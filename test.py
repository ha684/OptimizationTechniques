import ray


dataset = ray.data.read_csv()


trainer = ray.train.torch.TorchTrainer