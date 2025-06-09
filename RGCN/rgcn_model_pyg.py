# Core libs
import os
import argparse
import logging
import torch
import torch.optim as optim

# project modules
from model import RGCN
from data_utils import load_data, create_train_loader
from train_utils import train, extract_user_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Train RGCN model for link prediction")
    parser.add_argument("--data-path", type=str, default="../hetero_graph_v3.bin")
    parser.add_argument("--log-path", type=str, default="training.log")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--output-embeddings", action="store_true")
    return parser.parse_args()

def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

def main():
    args = parse_args()
    setup_logging(args.log_path)
    logging.info("Starting RGCN training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load and preprocess data
    data = load_data(args.data_path)
    # create data loaders
    train_loader = create_train_loader(data, args.batch_size, args.num_workers)
    # init model and optimizer
    model = RGCN(args.hidden_channels, args.out_channels, data.metadata(), dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # resume from checkpoint
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        model.load_state_dict(torch.load(args.ckpt_path))
        logging.info(f"Resumed from {args.ckpt_path}")
    # training loop
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch:02d} train_loss={loss:.4f}")
        torch.save(model.state_dict(), f"rgcn_epoch_{epoch}.pth")
        if args.output_embeddings and epoch % 2 == 0:
            out_path = f"user_emb_epoch_{epoch}.pt"
            extract_user_embeddings(model, data, device, args.batch_size, args.num_workers, args.out_channels, out_path)
    logging.info("Training complete")

# run
if __name__ == '__main__':
    main()