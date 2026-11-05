"""E_phi pretraining entry point.

Usage (from ReCoRe-main):
    python -m src.estimator.train --img_dir <M3FD_dir> --save_dir checkpoints/estimator
    python -m src.estimator.train --length 200 --epochs 2 --batch_size 4   # smoke test
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .network import DegradationEstimator
from .losses import compute_losses
from .data import DegradationPairDataset
from .calibrate import temperature_scale


def train(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    torch.manual_seed(args.seed)

    model = DegradationEstimator(
        n_classes=9, n_params=18, in_ch=1,
        descriptor_dim=args.descriptor_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    dataset = DegradationPairDataset(
        img_dir=args.img_dir, crop_size=args.crop_size,
        length=args.length, augment=not args.no_augment, seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (x, x_prime, active, norm) in enumerate(loader):
            x = x.to(device)
            x_prime = x_prime.to(device)
            active = active.to(device)
            norm = norm.to(device)
            optimizer.zero_grad()
            q_logit, phi, c_logit = model(x, x_prime)
            L_E, losses = compute_losses(q_logit, phi, c_logit, x, x_prime, active, norm)
            L_E.backward()
            optimizer.step()
            running += L_E.item()
        print("[epoch %d] L_E=%.4f  (L_cls=%.4f L_reg=%.4f L_syn=%.4f L_conf=%.4f)" % (
            epoch, running / max(len(loader), 1),
            losses["L_cls"].item(), losses["L_reg"].item(),
            losses["L_syn"].item(), losses["L_conf"].item()))
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "estimator_epoch%d.pt" % epoch))

    # temperature calibration
    T = 1.0
    if args.calibrate:
        cal_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        T = temperature_scale(model, cal_loader)
        print("calibrated temperature = %.4f" % T)
    torch.save({"state_dict": model.state_dict(), "temperature": T},
               os.path.join(args.save_dir, "estimator_final.pt"))
    print("saved to %s" % os.path.join(args.save_dir, "estimator_final.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=None, help="M3FD source image directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/estimator")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--length", type=int, default=1000, help="dataset length (samples in synthetic mode)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--descriptor_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()