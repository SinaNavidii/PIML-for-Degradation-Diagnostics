def test_PINN():
    # Load dummy data for testing
    X_train = torch.rand((1000, 100))  # Randomized data for training
    y_train = torch.rand((1000, 4))
    y_true_dd = torch.rand((1000, 4))
    X_test = torch.rand((200, 100))  # Randomized data for testing
    y_test = torch.rand((200, 4))

    # Instantiate the trainer
    trainer = PINNTrainer(
        input_size=100,
        hidden1=100,
        hidden2=10,
        output_size=4,
        surrogate_model_path='surrogate_model.pkl',
        half_cell_model_path='half_cell_model.pkl',
        learning_rate=0.002,
        batch_size=100,
        epochs=10,  # Reduced epochs for quick testing
        patience=5,
        scaling_factors=[1.0, 1.0, 1.0],
        seed=40
    )

    # Run training
    trainer.train(X_train, y_train, y_true_dd, X_test, y_test)
    print("Test run completed successfully.")


if __name__ == "__main__":
    test_PINN()
