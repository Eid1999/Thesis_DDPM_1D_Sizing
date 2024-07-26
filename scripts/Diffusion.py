import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Evaluations import test_performaces
from Evaluations.plot_gradient import plot_grad_flow


class DiffusionDPM:

    def __init__(
        self,
        noise_steps: int = 50,
        vect_size: int = 91,
        device: str = "cuda",
        X_norm_max: np.ndarray = np.array([1] * 12),
        X_norm_min: np.ndarray = np.array([-1] * 12),
    ) -> None:
        seed_value = 0
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
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
            beta_end: float = 0.02,
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
            # return betas * scaler
            return torch.clip(betas, 0.001, 0.999)

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
        epsilon = torch.randn_like(x)
        noisy_x = sqrt_alpha_hat * x + epsilon * sqrt_one_minus_alpha_hat
        return noisy_x, epsilon

    def sample_time(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.noise_steps, size=(n,), device=self.device)

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
            for i in reversed(range(0, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                alpha = self.alpha[t][:, None]
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
                beta = self.beta[t][:, None]
                z = (
                    torch.randn(
                        n, self.vect_size, 1, device=self.device, dtype=torch.float32
                    ).squeeze()
                    if i > 0
                    else torch.zeros_like(x)
                )
                uncoditional_predicted_noise = model(x, t)

                predicted_noise = model(x, t, y=y)
                predicted_noise = (
                    1 + weight
                ) * predicted_noise - weight * uncoditional_predicted_noise
                # predicted_noise=torch.lerp(uncoditional_predicted_noise,predicted_noise,weight)

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
        guidance_weight: float = 0.3,
        frequency_print: int = 50,
        visualise_grad: bool = False,
        noise_steps: int = 1000,
        data_type: str = "vcota",
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
            # weight_decay=1e-6,
        )
        Loss_Function = self.select_loss(loss_type)
        # Loss_Function= self.MAPE
        loss = 0
        if y is not None:
            dataloader = DataLoader(
                TensorDataset(x, y), batch_size=batch_size, shuffle=True
            )
        else:
            dataloader = DataLoader(
                TensorDataset(x), batch_size=batch_size, shuffle=True
            )
        os.makedirs(
            f"./weights/{data_type}/{type}/noise{self.noise_steps}", exist_ok=True
        )
        if trial == None:
            files = glob.glob(
                f"./weights/{data_type}/{type}/noise{self.noise_steps}/*.pth"
            )
            if len(files) != 0:
                for f in files:
                    os.remove(f)

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
                if visualise_grad and epoch % frequency_print == 0:
                    plot_grad_flow(self.model.named_parameters())
                optimizer.step()
                batch_loss_training.append(loss.item())
            if y_val is not None:
                self.model.eval()
                if epoch == epochs - 1 or epoch % frequency_print == 0:
                    error = test_performaces(
                        y_val,
                        self,
                        guidance_weight,
                        df_X,
                        df_y,
                        display=False,
                        data_type=data_type,
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
                    f"Performance Error: {np.mean(error):.4f}, Train Loss:{np.array(batch_loss_training[-1]):.4f}"
                )
                self.model.train()
            training_loss.append(np.mean(batch_loss_training))
            if trial is None and (
                (epoch) % frequency_print == 0 or epoch == epochs - 1
            ):
                torch.save(
                    self.model.state_dict(),
                    f"./weights/{data_type}/{type}/noise{self.noise_steps}/EPOCH{epoch+1}-PError: {np.mean(error):.4f}.pth",
                )
        del self.model
        if y_val is not None:
            return error.mean()
