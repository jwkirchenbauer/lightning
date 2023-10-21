import lightning as L
from lightning import Fabric
import torch
import torch.nn.functional as F

# from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split

from lightning.pytorch.strategies import FSDPStrategy
from lightning.fabric.strategies import FSDPStrategy


# Moved example to local due to need for patch to the demo transformer model definition
# https://github.com/Lightning-AI/lightning/discussions/14377
from transformer import Transformer, WikiText2

# swapping in dataset from huggingface
# from transformers import GPT2TokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from itertools import chain


def main(args):
    L.seed_everything(42)

    # torch.set_float32_matmul_precision("medium")

    # fsdp_strat = FSDPStrategy(
    #     sharding_strategy="FULL_SHARD",
    #     cpu_offload=True,
    # )
    fsdp_strat = FSDPStrategy()

    fabric = L.Fabric(
        accelerator=args.fabric_accelerator,
        # strategy=(args.fabric_strategy if args.fabric_strategy != "fsdp" else fsdp_strat),
        strategy=fsdp_strat,
        devices=args.fabric_devices,
        num_nodes=args.fabric_num_nodes,
        precision=args.fabric_precision,
    )
    fabric.print("Launching with args:", flush=True)
    for k, v in vars(args).items():
        # if k.startswith("fabric"):
        fabric.print(f"{k}:{v}", flush=True)
    fabric.launch()

    # Data
    # dataset = WikiText2()
    # train_dataloader, val_dataloader, _ = get_dataloaders(dataset, args)

    # swapping in dataset from huggingface
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size
    )

    # Model
    # dummy datasets
    # model = Transformer(vocab_size=dataset.vocab_size)
    # hf datasets and tokenizers
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
    # )
    # model.resize_token_embeddings(len(tokenizer))

    with fabric.init_module(empty_init=True):
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(len(tokenizer))

    fabric.barrier()
    print("Setting up modules...")

    # Optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    # works for ddp
    # model, optimizer = fabric.setup(model, optimizer)

    # for fsdp
    model = fabric.setup_module(model)
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    train(fabric, model, optimizer, train_dataloader, val_dataloader)


def train(fabric, model, optimizer, train_dataloader, val_dataloader, max_epochs=20):
    for epoch in range(max_epochs):
        train_epoch(fabric, model, optimizer, train_dataloader, epoch)
        val_loss = validate(fabric, model, val_dataloader)
        fabric.print(f"val loss {val_loss.item():.4f}")


def train_epoch(fabric, model, optimizer, train_dataloader, epoch):
    for batch_idx, batch in enumerate(train_dataloader):
        # toy transformer and dataset version
        # input, target = batch
        # output = model(input, target)
        # loss = F.nll_loss(output, target.view(-1))

        # hf datasets version, toy transformer
        # output = model(batch["input_ids"], batch["labels"])
        # loss = F.nll_loss(output, batch["labels"])

        # hf datasets version, hf transformer
        if batch["input_ids"].shape[-1] == 0:
            continue
        # print(batch)
        outputs = model(**batch)
        loss = outputs.loss

        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, clip_val=0.25)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 200 == 0:
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - loss {loss.item():.4f}")


@torch.no_grad()
def validate(fabric, model, val_dataloader):
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(len(val_dataloader))
    for k, batch in enumerate(val_dataloader):
        # toy transformer and dataset version
        # input, target = batch
        # output = model(input, target)
        # loss = F.nll_loss(output, target.view(-1))
        # hf datasets version, hf transformer
        if batch["input_ids"].shape[-1] == 0:
            continue
        # print(batch)
        outputs = model(**batch)
        loss = outputs.loss

        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_dataloaders(dataset, args):
    n = len(dataset)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n - 4000, 2000, 2000], generator=generator
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fabric_accelerator", type=str, default="cuda")
    parser.add_argument("--fabric_strategy", type=str, default="ddp")
    parser.add_argument("--fabric_devices", type=int, default=1)
    parser.add_argument("--fabric_num_nodes", type=int, default=1)
    parser.add_argument("--fabric_precision", type=str, default="32-true")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    args = parser.parse_args()

    main(args)
