import logging
import sys
import os
from math import ceil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from monai.networks.nets import UNet
from monai.losses import DiceLoss

import monai

from dataset import partition_train_dataset, partition_validation_dataset


checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')


def setup(rank, world_size):
    os.environ['MASTER_PORT'] = '29500'

    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    print(f"Training on rank {rank}.")
    setup(rank, world_size)

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fun = DiceLoss(
        sigmoid=True
    )

    optimizer = Adam(ddp_model.parameters(), lr=1e-4)

    N_EPOCHES = 100

    train_set, train_batch_size, train_sampler = partition_train_dataset()
    validation_set, validation_batch_size, validation_sampler = partition_validation_dataset()

    best_validation_loss = float('inf')
    best_validation_loss_epoch = -1
    for epoch in range(N_EPOCHES):
        print("=" * 20)
        print(f"EPOCH {epoch + 1}/{N_EPOCHES}")

        train_sampler.set_epoch(epoch)
        
        train_num_batches = ceil(len(train_sampler) / float(train_batch_size))

        ddp_model.train()
        epoch_train_loss = 0
        for i, train_batch in enumerate(train_set):
            train_imgs, train_masks = train_batch['img'].to(rank), train_batch['mask'].to(rank)
            optimizer.zero_grad()
            pred = ddp_model(train_imgs)
            loss = loss_fun(pred, train_masks)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            print(f"\t{i}/{train_num_batches} train_loss: {loss.item():.4f}")
        epoch_train_loss /= train_num_batches
        print(f"\tAverage train loss: {epoch_train_loss:.4f}")

        ddp_model.eval()
        with torch.no_grad():
            validation_num_batches = ceil(len(validation_sampler) / float(validation_batch_size))

            epoch_validation_loss = 0
            for validation_batch in validation_set:
                validation_imgs, validation_masks = validation_batch['img'].to(rank), validation_batch['mask'].to(rank)
                pred = ddp_model(validation_imgs)
                loss = loss_fun(pred, validation_masks)
                epoch_validation_loss += loss.item()
            epoch_validation_loss /= validation_num_batches
            print(f"\tAverage validation loss: {epoch_validation_loss:.4f}")

            # ADD EARLY STOPPING

            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                best_validation_loss_epoch = epoch + 1
                if rank == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_train_loss,
                        'validation_loss': epoch_validation_loss,
                    }, os.path.join(checkpoint_dir, 'best.pth'))

        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'validation_loss': epoch_validation_loss,
            }, os.path.join(checkpoint_dir, 'last.pth'))

    print(f"TRAIN COMPLETED!")
    print(f"best_metric: {best_validation_loss:.4f} at epoch: {best_validation_loss_epoch}")

    cleanup()
    print(f"Finished training on rank {rank}.")


if __name__ == "__main__":
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    os.makedirs(checkpoint_dir, exist_ok=True)

    n_gpus = torch.accelerator.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
