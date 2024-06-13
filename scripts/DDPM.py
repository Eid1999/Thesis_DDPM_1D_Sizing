from requirements import *
from scripts.Evaluations import test_performaces


torch.manual_seed(0)


class Diffusion:
    def __init__(
        self,
        noise_steps=500,
        vect_size=91,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.vect_size = vect_size
        self.device = device
        self.X_norm_max = torch.tensor(
            [1] * 12, device=self.device, dtype=torch.float32
        )
        self.X_norm_min = torch.tensor(
            [-1] * 12, device=self.device, dtype=torch.float32
        )

        self.beta = self.prepare_noise_schedule().to(device)
        # self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(
        self,
        scheduler="cos",
    ):

        # linear schedule
        def linear_schedule(beta_start=0.0001, beta_end=0.01):

            return torch.linspace(beta_start, beta_end, self.noise_steps)

        # cosine schedule
        def cosine_schedule(
            scaler=1,
            s=0.008,
        ):
            x = torch.linspace(0, self.noise_steps, self.noise_steps + 1)
            alphas_cumprod = (
                torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.99999) * scaler

        def logarithmic_schedule(beta_start=0.0001, beta_end=0.02):
            return torch.logspace(
                start=beta_start,
                end=beta_end,
                steps=self.noise_steps,
            )

        noise_scheduler = {
            "cos": cosine_schedule,
            "log": logarithmic_schedule,
            "lin": linear_schedule,
        }
        return noise_scheduler[scheduler]()

    def forward_process(self, x, t):
        # p(xt|x0) := N (xt; √αtx0, (1 − αtI))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha[t])[:, None]
        epsilon = torch.rand_like(x)
        noisy_x = sqrt_alpha_hat * x + epsilon * sqrt_one_minus_alpha_hat
        return noisy_x, epsilon

    def sample_time(self, n):
        return torch.randint(0, self.noise_steps, size=(n,), device=self.device)

    def sampling(self, model, n, y, weight=1.0):
        model.eval()

        with torch.no_grad():
            x = torch.randn(
                n, self.vect_size, 1, device=self.device, dtype=torch.float
            ).squeeze()
            for i in reversed(range(0, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                alpha = self.alpha[t][:, None]
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
                beta = self.beta[t][:, None]
                z = (
                    torch.randn(
                        n, self.vect_size, 1, device=self.device, dtype=torch.float32
                    ).squeeze()
                    if i != 0
                    else torch.zeros_like(x)
                )
                uncoditional_predicted_noise = model(x, t)
                predicted_noise = 0
                if weight > 0:
                    predicted_noise = model(x, t, y=y)
                    predicted_noise = (1 + weight) * predicted_noise
                    # predicted_noise=torch.lerp(uncoditional_predicted_noise,predicted_noise,weight)
                predicted_noise -= weight * uncoditional_predicted_noise
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / sqrt_one_minus_alpha_hat) * predicted_noise)
                    + torch.sqrt(beta) * z
                )
        x = torch.minimum(torch.maximum(x, self.X_norm_min), self.X_norm_max)

        model.train()

        return x

    def select_loss(self, type):
        if type == "l2":
            return nn.MSELoss(reduction="mean")
        elif type == "l1":
            return nn.L1Loss(reduction="mean")
        elif type == "huber":
            return nn.HuberLoss(reduction="mean")
        else:
            return nn.HuberLoss(reduction="mean")

    def reverse_process(
        self,
        x,
        y,
        network,
        df_X,
        df_y,
        epochs=1000,
        batch_size=32,
        X_val=None,
        y_val=None,
        hidden_layers=[80],
        learning_rate=1e-5,
        loss_type="l2",
        early_stop=True,
        trial=None,
        guidance_weight=20,
    ):
        self.model = network(
            input_size=x.shape[1],
            output_size=x.shape[1],
            y_dim=y.shape[-1],
            hidden_layers=hidden_layers,
        ).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        Loss_Function = self.select_loss(loss_type)
        # Loss_Function= self.MAPE
        loss = 0
        dataloader = DataLoader(
            TensorDataset(x, y), batch_size=batch_size, shuffle=False
        )

        training_loss = []
        val_loss = []
        best_val_loss = float("inf")
        counter = 0
        self.model.eval()
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            batch_loss_training = []

            for i, (vector, y) in enumerate(dataloader):
                vector = vector.to(self.device)
                t = self.sample_time(vector.shape[0]).to(self.device)
                vector, noise = self.forward_process(vector, t)
                if np.random.random() < 0.1:
                    y = None
                predicted_noise = self.model(vector, t, y)
                loss = Loss_Function(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss_training.append(loss.item())
            if y_val is not None:
                self.model.eval()
                if False or epoch % 50 == 0:
                    error = test_performaces(
                        y_val,
                        self,
                        guidance_weight,
                        df_X,
                        df_y,
                        display=False,
                    )

                    if early_stop:
                        if np.mean(error) < (best_val_loss):
                            best_val_loss = np.mean(error)
                            counter = 0
                        else:
                            counter += 10
                            if counter >= 10:
                                print(f"Early stop at epoch {epoch + 1}")
                                break
                    if trial is not None:
                        trial.report(np.mean(error), step=epoch + 1)
                        # if trial.should_prune():
                        # raise optuna.exceptions.TrialPruned()
                pbar.set_description(f"Performance Error: {np.mean(error)}")
                self.model.train()
            training_loss = np.mean(batch_loss_training)

        return training_loss, error if y_val is not None else training_loss
