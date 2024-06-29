from libraries import *
from Evaluations import test_performaces
from Evaluations.plot_gradient import plot_grad_flow


torch.manual_seed(0)


class DiffusionDPM:

    def __init__(
        self,
        noise_steps: int = 50,
        vect_size: int = 91,
        device: str = "cuda",
        X_norm_max: np.ndarray = np.array([1] * 12),
        X_norm_min: np.ndarray = np.array([1] * 12),
    ) -> None:
        self.noise_steps = noise_steps
        self.vect_size = vect_size
        self.device = device
        self.X_norm_max = torch.tensor(
            X_norm_max,
            device=self.device,
            dtype=torch.float32,
        )
        self.X_norm_min = torch.tensor(
            X_norm_min,
            device=self.device,
            dtype=torch.float32,
        )

        self.beta = self.prepare_noise_schedule().to(device)
        # self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(
        self,
        scheduler: str = "cos",
    ) -> torch.Tensor:

        # linear schedule
        def linear_schedule(
            beta_start: float = 0.0001,
            beta_end: float = 0.01,
        ) -> torch.Tensor:

            return torch.linspace(beta_start, beta_end, self.noise_steps)

        # cosine schedule
        def cosine_schedule(
            s: float = 0.008,
        ) -> torch.Tensor:
            x = torch.linspace(0, self.noise_steps, self.noise_steps + 1)
            alphas_cumprod = (
                torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.99999)

        def logarithmic_schedule(
            beta_start: float = 0.0001, beta_end: float = 0.02
        ) -> torch.Tensor:
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

    def forward_process(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # p(xt|x0) := N (xt; √αtx0, (1 − αtI))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha[t])[:, None]
        epsilon = torch.rand_like(x)
        noisy_x = sqrt_alpha_hat * x + epsilon * sqrt_one_minus_alpha_hat
        return noisy_x, epsilon

    def sample_time(self, n: int) -> torch.Tensor:
        return torch.randint(1, self.noise_steps, size=(n,), device=self.device)

    def sampling(
        self,
        model: nn.Module,
        n: int,
        y: torch.Tensor,
        weight: float = 1.0,
    ) -> torch.Tensor:
        model.eval()

        with torch.no_grad():
            x = torch.randn(
                n, self.vect_size, 1, device=self.device, dtype=torch.float
            ).squeeze()
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                alpha = self.alpha[t][:, None]
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
                beta = self.beta[t][:, None]
                z = (
                    torch.randn(
                        n, self.vect_size, 1, device=self.device, dtype=torch.float32
                    ).squeeze()
                    if i > 1
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

    def select_loss(self, type: str) -> nn.Module:
        if type == "mse":
            return nn.MSELoss(reduction="mean")
        elif type == "l1":
            return nn.L1Loss(reduction="mean")
        elif type == "huber":
            return nn.HuberLoss(reduction="mean")
        else:
            return nn.HuberLoss(reduction="mean")

    def reverse_process(
        self,
        x: torch.Tensor,
        y: Union[torch.Tensor, None],
        network: Type[nn.Module],
        df_X: pd.DataFrame,
        df_y: pd.DataFrame,
        type: str,
        epochs: int = 1000,
        batch_size: int = 32,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        nn_template: dict = {"hidden_layer": 80},
        learning_rate: float = 1e-5,
        loss_type: str = "mse",
        early_stop: bool = True,
        trial: Optional[Trial] = None,
        guidance_weight: int = 20,
        frequency_print: int = 50,
        visualise_grad: bool = False,
        noise_steps: int = 1000,
    ) -> Optional[float]:
        self.model = network(
            input_size=x.shape[1],
            output_size=x.shape[1],
            y_dim=y.shape[-1] if y is not None else None,
            **nn_template,
        ).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        Loss_Function = self.select_loss(loss_type)
        # Loss_Function= self.MAPE
        loss = 0
        if y is not None:
            dataloader = DataLoader(
                TensorDataset(x, y), batch_size=batch_size, shuffle=False
            )
        else:
            dataloader = DataLoader(
                TensorDataset(x), batch_size=batch_size, shuffle=False
            )
        if trial == None:
            files = glob.glob(f"./weights/{type}/noise{self.noise_steps}/*.pth")
            if len(files) != 0:
                for f in files:
                    os.remove(f)

        training_loss = []
        val_loss = []
        best_val_loss = float("inf")
        counter = 0
        self.model.eval()
        pbar = tqdm(range(epochs))
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        os.makedirs(f"./weights/{type}/noise{self.noise_steps}", exist_ok=True)
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
                if visualise_grad and epoch % frequency_print == 0:
                    plot_grad_flow(self.model.named_parameters())
                optimizer.step()
                scheduler.step()
                batch_loss_training.append(loss.item())
            if y_val is not None:
                self.model.eval()
                if False or epoch % frequency_print == 0:
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
                pbar.set_description(
                    f"Performance Error: {np.mean(error):.4f}, Train Loss:{np.array(batch_loss_training[-1]).max():.4f}"
                )
                self.model.train()
            training_loss.append(np.mean(batch_loss_training))
            if trial is None and (epoch) % frequency_print == 0:
                torch.save(
                    self.model.state_dict(),
                    f"./weights/{type}/noise{self.noise_steps}/EPOCH{epoch}-PError: {np.mean(error):.4f}.pth",
                )
        torch.save(
            self.model.state_dict(),
            f"./weights/{type}/noise{self.noise_steps}/EPOCH{epochs}.pth",
        )
        if y_val is not None:
            return error.mean()
