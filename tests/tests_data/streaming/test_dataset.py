# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from unittest import mock

import pytest
from lightning import seed_everything
from lightning.data.datasets.env import _DistributedEnv
from lightning.data.streaming import Cache
from lightning.data.streaming.dataset import StreamingDataset, _try_create_cache_dir
from lightning.data.streaming.shuffle import FullShuffle, NoShuffle
from torch.utils.data import DataLoader


def test_streaming_dataset(tmpdir, monkeypatch):
    seed_everything(42)

    dataset = StreamingDataset(input_dir=str(tmpdir))
    with pytest.raises(ValueError, match="The provided dataset"):
        iter(dataset)
    dataset = StreamingDataset(input_dir=str(tmpdir))
    with pytest.raises(ValueError, match="The provided dataset"):
        _ = dataset[0]

    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(12):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))

    assert len(dataset) == 12
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 12

    dataloader = DataLoader(dataset, num_workers=2, batch_size=1)
    assert len(dataloader) == 12
    dataloader = DataLoader(dataset, num_workers=2, batch_size=2)
    assert len(dataloader) == 6


@mock.patch.dict(os.environ, {"LIGHTNING_CLUSTER_ID": "123", "LIGHTNING_CLOUD_PROJECT_ID": "456"})
@mock.patch("lightning.data.streaming.dataset.os.makedirs")
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_create_cache_dir_in_lightning_cloud(makedirs_mock):
    # Locally, we can't actually write to the root filesystem with user privileges, so we need to mock the call
    dataset = StreamingDataset("dummy")
    with pytest.raises(FileNotFoundError, match="/0` doesn't exist"):
        iter(dataset)
    makedirs_mock.assert_called()


@pytest.mark.parametrize("drop_last", [False, True])
def test_streaming_dataset_distributed_no_shuffle(drop_last, tmpdir):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(101):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False, drop_last=drop_last)
    assert not dataset.shuffle
    _ = dataset[0]  # init shuffler
    assert isinstance(dataset.shuffler, NoShuffle)

    for i in range(101):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(1, 0)
    assert len(dataset) == 101

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 50 + int(not drop_last)
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50 + int(not drop_last)
    process_1_1 = list(dataset_iter)
    assert len(process_1_1) == 50 + int(not drop_last)
    assert process_1_1[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50 + int(not drop_last)
    process_1_2 = list(dataset_iter)
    assert process_1_2[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert len(process_1_2) == 50 + int(not drop_last)

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False, drop_last=drop_last)
    dataset.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset) == 50
    dataset_iter = iter(dataset)
    process_2_1 = list(dataset_iter)
    assert process_2_1[:10] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert len(process_2_1) == 50
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_2_2 = list(dataset_iter)
    assert process_2_2[:10] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    assert len(process_2_2) == 50

    _, intervals_per_ranks = dataset.shuffler.get_chunks_and_intervals_per_ranks(
        dataset.distributed_env, dataset.current_epoch
    )

    assert process_1_1 == process_1_2

    found_list = []
    for i in process_1_1:
        found = False
        for interval in intervals_per_ranks[0]:
            if interval[0] <= i <= interval[1]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    found_list = []
    for i in process_2_1:
        found = False
        for interval in intervals_per_ranks[1]:
            if interval[0] <= i <= interval[1]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    assert len([i for i in process_1_1 if i in process_2_1]) == 0
    assert len([i for i in process_1_2 if i in process_2_2]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
def test_streaming_dataset_distributed_full_shuffle_odd(drop_last, tmpdir):
    seed_everything(42)

    cache = Cache(input_dir=str(tmpdir), chunk_size=10)
    for i in range(1097):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1097):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 548
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 548
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [785, 788, 782, 783, 789, 787, 786, 781, 784, 780]
    assert len(process_1_1) == 548

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset_2) == 548 + int(not drop_last)
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 548 + int(not drop_last)
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [939, 938, 252, 259, 257, 255, 258, 253, 250, 251]
    assert len(process_2_1) == 548 + int(not drop_last)
    assert len([i for i in process_1_1 if i in process_2_1]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
def test_streaming_dataset_distributed_full_shuffle_even(drop_last, tmpdir):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(1222):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1222):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 611
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 611
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [185, 184, 182, 189, 187, 181, 183, 180, 186, 188]
    assert len(process_1_1) == 611

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset_2) == 611
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 611
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [813, 815, 816, 812, 818, 811, 817, 814, 819, 277]
    assert len(process_2_1) == 611

    assert len([i for i in process_1_1 if i in process_2_1]) == 0


def test_streaming_dataset_deepcopy(tmpdir, monkeypatch):
    seed_everything(42)

    remote_dir = os.path.join(tmpdir, "remote_dir")

    os.makedirs(remote_dir, exist_ok=True)

    cache = Cache(remote_dir, chunk_size=10)
    for i in range(10):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=remote_dir, shuffle=True)
    assert dataset.cache is None
    iter(dataset)
    assert dataset.cache is not None
    assert dataset.cache._reader._prepare_thread is None
    dataset.cache._reader._prepare_thread = True
    dataloader = DataLoader(dataset, num_workers=1)

    batches = []
    for batch in dataloader:
        batches.append(batch)

    assert len(batches) == 10


def test_dataset_cache_recreation(tmpdir):
    """Test that we recreate the cache and other objects only when appropriate."""
    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(10):
        cache[i] = i
    cache.done()
    cache.merge()

    # repated `len()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    len(dataset)
    assert not dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(shuffler, NoShuffle)
    len(dataset)
    assert dataset.shuffler is shuffler

    # repeated `iter()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    iter(dataset)
    cache = dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(cache, Cache)
    assert isinstance(shuffler, NoShuffle)
    iter(dataset)
    assert isinstance(dataset.cache, Cache)
    assert isinstance(dataset.shuffler, NoShuffle)
    assert dataset.cache is not cache  # cache gets recreated
    assert dataset.shuffler is not shuffler  # shuffler gets recreated

    # repeated `getitem()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    _ = dataset[0]
    cache = dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(cache, Cache)
    assert isinstance(shuffler, NoShuffle)
    _ = dataset[1]
    assert dataset.cache is cache  # cache gets reused
    assert dataset.shuffler is shuffler  # shuffler gets reused


def test_try_create_cache_dir():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _try_create_cache_dir("any") is None

    # the cache dir creating at /cache requires root privileges, so we need to mock `os.makedirs()`
    with (
        mock.patch.dict("os.environ", {"LIGHTNING_CLUSTER_ID": "abc", "LIGHTNING_CLOUD_PROJECT_ID": "123"}),
        mock.patch("lightning.data.streaming.dataset.os.makedirs") as makedirs_mock,
    ):
        cache_dir_1 = _try_create_cache_dir("")
        cache_dir_2 = _try_create_cache_dir("ssdf")
        assert cache_dir_1 != cache_dir_2
        assert cache_dir_1 == os.path.join("/cache", "chunks", "d41d8cd98f00b204e9800998ecf8427e", "0")
        assert len(makedirs_mock.mock_calls) == 2

        assert _try_create_cache_dir("dir", shard_rank=0) == os.path.join(
            "/cache", "chunks", "736007832d2167baaae763fd3a3f3cf1", "0"
        )
        assert _try_create_cache_dir("dir", shard_rank=1) == os.path.join(
            "/cache", "chunks", "736007832d2167baaae763fd3a3f3cf1", "1"
        )
        assert _try_create_cache_dir("dir", shard_rank=2) == os.path.join(
            "/cache", "chunks", "736007832d2167baaae763fd3a3f3cf1", "2"
        )
