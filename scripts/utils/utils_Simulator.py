from libraries import *
from Networks import Simulator


def Eval_loop(
    X: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    return loss.cpu().numpy()


def Train_loop(
    data_loader_train: DataLoader,
    model: nn.Module,
    optimizer: optim.Adam,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    model.train()
    for X, y in data_loader_train:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def epoch_loop(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    nn_template: dict = {"hidden_layers": [10]},
    batch_size: int = 32,
    lr: float = 1e-5,
    n_epoch: int = 100,
    trial: Optional[Trial] = None,
) -> tuple[nn.Module, float]:
    last_val_loss = float("inf")
    model = Simulator(
        X_train.shape[-1],
        y_train.shape[-1],
        **nn_template,
    ).to("cuda")
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-7,
    )
    patience = 0
    dataloader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    loss_fn = nn.L1Loss(reduction="mean")
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    val_loss = Eval_loop(X_val, y_val, model, loss_fn)
    pbar = tqdm(range(n_epoch))
    for epoch in pbar:
        pbar.set_description(f"Validarion Loss: {val_loss}")
        Train_loop(dataloader, model, optimizer, loss_fn)
        scheduler.step()
        if trial is not None:
            trial.report(val_loss, step=epoch + 1)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        val_loss = Eval_loop(X_val, y_val, model, loss_fn)
        if val_loss < last_val_loss:
            last_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience == 10:
                break
    return model, val_loss
