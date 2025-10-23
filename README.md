# Sign Language Recognition Application

This application is a real-time sign language recognition system that uses computer vision and deep learning to interpret sign language gestures. It can recognize both alphabets and common phrases in sign language using your computer's webcam.

## Features

- Real-time hand gesture recognition
- Support for alphabet signs (A-Z)
- Common phrase recognition
- Live webcam feed with visual feedback
- Gesture prediction with confidence scores
- User-friendly interface

## Prerequisites

- Python 3.10 or higher
- Webcam
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd sign_language_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv signlang_env
source signlang_env/bin/activate  # On Windows: signlang_env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. To run the main application:
```bash
python modified.py
```

2. To collect new training data:
```bash
python collect_images.py
```

3. To train the model:
```bash
python training.py
```

## Project Structure

- `modified.py`: Main application file for real-time sign language recognition
- `data.py`: Data processing and preparation utilities
- `training.py`: Model training script
- `collect_images.py`: Script for collecting new training data
- `*.h5`: Trained model files
- `*.pkl`: Label encoder and other serialized files

## Model Information

The application uses a deep learning model trained on hand gesture data. It utilizes:
- MediaPipe for hand landmark detection
- TensorFlow/Keras for the neural network model
- OpenCV for image processing and webcam interface

## Dataset

The dataset includes:
- Alphabet signs (A-Z)
- Common phrases
- Multiple samples per gesture
- Various hand positions and lighting conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.