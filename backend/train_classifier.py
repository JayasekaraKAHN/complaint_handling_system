from app.classifier import train_classifier
from app import config

if __name__ == "__main__":
    print("Looking for complaints file at:", config.COMPLAINTS_DATA)
    model, categories = train_classifier(config.COMPLAINTS_DATA)
    print("Training complete. Categories learned:", categories)
