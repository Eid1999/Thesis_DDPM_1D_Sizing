from requirements import *
from Networks import Simulator


def Eval_loop(X, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    return loss


def Train_loop(data_loader_train, model, optimizer, loss_fn):
    model.train()
    for X, y in data_loader_train:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def epoch_loop(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layers=[10],
    batch_size=32,
    lr=1e-5,
    n_epoch=100,
    trial=None,
):
    last_val_loss = float("inf")
    model = Simulator(
        X_train.shape[-1],
        y_train.shape[-1],
        hidden_layers=hidden_layers,
    ).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    patience = 0
    dataloader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    loss_fn = nn.MSELoss(reduction="mean")
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    val_loss = Eval_loop(X_val, y_val, model, loss_fn)
    pbar = tqdm(range(n_epoch))
    for epoch in pbar:
        pbar.set_description(f"Validarion Loss: {val_loss}")
        Train_loop(dataloader, model, optimizer, loss_fn)
        scheduler.step()
        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        val_loss = Eval_loop(X_val, y_val, model, loss_fn)
        if val_loss < last_val_loss:
            last_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience == 5:
                break
    return model, val_loss
