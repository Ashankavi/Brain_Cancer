# මුලින්ම අපි අපිට ඕන කරන හැම library එකක්ම import කරගමු.
# This imports the necessary libraries
import torch # PyTorch තමයි මේ project එකේ ප්‍රධානම library එක. මේකෙන් තමයි අපි deep learning model එක හදලා train කරන්නේ.
import torch.nn as nn # Model එක හදන්න ඕන කරන layers (උදා: Linear, Conv2d) වගේ දේවල් හදාගන්න මේක උදව් වෙනවා.
from torchvision import datasets, transforms, models # PyTorch වලම තවත් කොටසක්. images එක්ක වැඩ කරන්න ගොඩක් උදව් වෙනවා.
from torchvision.models import ResNet50_Weights # ResNet50 කියලා කලින් හදලා train කරපු model එකක weights (දත්ත) ටික load කරගමු.
from torch.utils.data import DataLoader, random_split # Data batches වලට කඩන්න, ඒ වගේම dataset එක training and validation sets වලට වෙන් කරන්න මේක පාවිච්චි කරනවා.
import matplotlib.pyplot as plt # Training යන අතරතුර loss සහ accuracy graphs බලන්න මේකෙන් පුළුවන්.
import os # Files සහ folders එක්ක වැඩ කරන්න අවශ්‍යයි.

# Model එක train කරන ප්‍රධාන කොටස (function) මෙතන තියෙනවා.
def train_model():
    """
    මොලයේ පිළිකා වර්ගීකරණය සඳහා ResNet50 model එකක් train කර හොඳම model එක save කරනවා.
    Trains a ResNet50 model for brain cancer detection and saves the best model.
    """

    # Model එක save කරන file එකේ නම සහ training වලට අවශ්‍ය අනිත් දේවල් මෙතන තියෙනවා.
    best_model_filename = "brain_cancer_model.pth"
    best_val_accuracy = 0.0 # හොඳම validation accuracy එක තියාගමු.
    patience_counter = 0 # Early stopping කරන්න counter එකක් තියාගමු.
    patience = 5 # Validation accuracy එක වැඩි වුණේ නැත්නම් training එක නවත්වන්න epochs කීයක් ඉවසනවද කියලා මෙතන තියෙනවා.

    # Images train කරන්න කලින් සුදානම් කරගමු.
    # Images model එකට දෙන්න කලින් අවශ්‍ය විදියට වෙනස් කරගමු.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Images ඔක්කොම 224x224 size එකට හරවමු.
        transforms.RandomHorizontalFlip(p=0.5), # Images අහඹු විදියට දෙපැත්තට හරවමු. මේකෙන් model එකට විවිධ images හඳුනාගන්න පුළුවන්.
        transforms.RandomRotation(15), # Images ටිකක් කරකවමු.
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Images වල brightness සහ contrast ටිකක් වෙනස් කරමු.
        transforms.ToTensor(), # Image එක PyTorch tensor එකක් බවට convert කරගමු.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Image එක normalize කරමු.
    ])

    try:
        # 'data' folder එකේ තියෙන images load කරගමු.
        # Folders වලට අනුව images classes වලට බෙදලා තියෙනවා.
        full_dataset = datasets.ImageFolder(root='data', transform=train_transform)
    except FileNotFoundError as e:
        print("Error: Dataset not found. Please ensure the directory structure is correct.")
        print("The expected structure is 'data/brain_glioma', 'data/brain_menin', etc.")
        return

    # Dataset එක 80% training set එකටත්, 20% validation set එකටත් බෙදමු.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data loader එක හදලා batches වලට data බෙදලා model එකට දෙමු.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {full_dataset.classes}")

    # Model එක train කරන්න GPU (cuda) එකක් තියෙනවද කියලා බලමු.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ResNet50 කියලා කලින් හදපු model එක load කරගමු.
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Model එකේ layers freeze කරමු. ඒ කියන්නේ ඒ layers වල weights training එකේදී වෙනස් වෙන්නේ නැහැ.
    for param in model.parameters():
        param.requires_grad = False

    # Model එකේ අන්තිම layer එක අපේ classes ගණනට අනුව වෙනස් කරමු.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device) # Model එක GPU එකට යවමු.

    # Loss function එක සහ optimizer එක define කරමු.
    # Loss function එකෙන් model එකේ වැරදි ගණනය කරනවා.
    criterion = nn.CrossEntropyLoss()
    # Optimizer එකෙන් weights update කරලා වැරදි අඩු කරනවා.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    # Learning rate scheduler එක. accuracy එක වැඩි වුණේ නැත්නම් learning rate එක අඩු කරනවා.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    num_epochs = 50 # Epochs ගණන define කරමු.
    train_losses = [] # Training loss එක තියාගමු.
    val_losses = [] # Validation loss එක තියාගමු.
    train_accuracies = [] # Training accuracy එක තියාගමු.
    val_accuracies = [] # Validation accuracy එක තියාගමු.

    # Training loop එක පටන් ගමු.
    for epoch in range(num_epochs):
        model.train() # Model එක training mode එකට දාමු.
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training data batches වශයෙන් load කරගමු.
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # Gradients reset කරමු.
            outputs = model(images) # Prediction එක කරමු.
            loss = criterion(outputs, labels) # Loss එක calculate කරමු.
            loss.backward() # Backpropagation කරලා gradients calculate කරමු.
            optimizer.step() # Weights update කරමු.
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Epoch එක අවසානයේ training metrics calculate කරමු.
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase එක පටන් ගමු.
        model.eval() # Model එක evaluation mode එකට දාමු.
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Gradient ගණනය කරන්නේ නැති බව කියනවා. (memory save කරගන්න).
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Epoch එක අවසානයේ validation metrics calculate කරමු.
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        scheduler.step(val_acc) # Learning rate scheduler එක update කරමු.

        # හොඳම model එක save කරගමු.
        if val_acc > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.2f}% to {val_acc:.2f}%. Saving model...")
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_filename)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Training එක අවසානයේ හොඳම model එක load කරගමු.
    if os.path.exists(best_model_filename):
        model.load_state_dict(torch.load(best_model_filename, map_location=device))
        print(f"Loaded best model '{best_model_filename}' for final evaluation.")
    else:
        print(f"Warning: Best model file '{best_model_filename}' not found. Using the last trained model.")

    # Training සහ validation graphs plot කරමු.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    print("\nTraining complete.")


# මේ script එක run කරද්දී train_model() function එක run වෙනවා.
if __name__ == '__main__':
    train_model()
