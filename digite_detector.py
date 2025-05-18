def predict_image(image_path):
    model = DigitCNN()
    model.load_state_dict(torch.load("digit_detector.pth"))
    model.eval()

    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.1307,), (0.3081,))(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print("Predicted Digit:", predicted.item())

# Example
predict_image("my_digit.png")
