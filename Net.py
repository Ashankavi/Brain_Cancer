import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A basic CNN model for multi-class classification.
    """

    def __init__(self, num_classes):
        # මේ function එකෙන් තමයි model එකේ layers හදන්නේ.
        super(SimpleCNN, self).__init__()

        # මේක තමයි image එක analyze කරන ප්‍රධාන layer එක.
        # මේකේදී image එකේ තියෙන විශේෂ features (හැඩතල, colours වගේ) හොයාගන්නවා.
        self.conv_layer = nn.Sequential(
            # මේ layer එකෙන් image එකේ patterns analyze කරලා අලුත් features හදනවා.
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # මේකෙන් negative values ඔක්කොම 0 කරනවා.
            nn.ReLU(),
            # මේකෙන් image එකේ size එක අඩු කරනවා. ඒකෙන් අනවශ්‍ය details අයින් වෙනවා.
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # මේක තමයි image එක classify කරන ප්‍රධාන layer එක.
        # මේකේදී කලින් හොයාගත් features වලින් image එක මොකක්ද කියලා තීරණය කරනවා.
        self.fc_layer = nn.Sequential(
            # මේකෙන් කලින් layers වලින් ආපු data ඔක්කොම එක තීරුවකට හරවගන්නවා.
            nn.Flatten(),
            # මේ layer එකෙන් data එක analyze කරලා වර්ග කරන්න සූදානම් කරනවා.
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            # මේක තමයි අන්තිම layer එක. මේකෙන් image එක අදාල class එකට classify කරනවා.
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # මේ function එකෙන් තමයි image එක layers හරහා යවන්නේ.
        # conv_layer එකෙන් features extract කරලා, fc_layer එකට යවනවා classify කරන්න.
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
