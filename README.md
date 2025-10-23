# Sign Language Detection System

## Overview
This project implements a real-time sign language detection system using computer vision and deep learning. It can recognize both alphabets (A-Z) and common phrases in sign language through webcam input.

## Key Features
- Real-time hand gesture recognition
- Supports both alphabet signs (A-Z) and common phrases
- Live webcam feed with visual feedback
- High accuracy in gesture prediction
- User-friendly interface
- Support for multiple gestures including:
  - Alphabets (A to Z)
  - Common phrases ("Hello", "Thank You", "I Love You", "Yes", "No")

## Technology Stack
- Python 3.10+
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- MediaPipe for hand landmark detection
- NumPy for numerical computations
- Scikit-learn for machine learning utilities

## Project Structure
```
sign_language_detector/
├── Alphabet_Data/         # Training data for alphabets
├── MP_Data/              # Training data for phrases
├── static/              # Static files (CSS)
├── templates/           # HTML templates
├── app.py              # Main Flask application
├── collect_images.py   # Script for collecting training images
├── data.py            # Data processing utilities
├── hand_tracking.py   # Hand tracking implementation
├── modified.py        # Modified implementation
├── motion_capture.py  # Motion capture functionality
├── real_time_detect.py # Real-time detection script
├── train_model.py     # Model training script
├── training.py        # Training utilities
└── requirements.txt   # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dilip-ravichandra/sign_language_detector.git
cd sign_language_detector
```

2. Create and activate virtual environment:
```bash
python -m venv signlang_env
# On Windows:
signlang_env\Scripts\activate
# On Unix or MacOS:
source signlang_env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python modified.py
```

2. Collect new training data:
```bash
python collect_images.py
```

3. Train the model:
```bash
python training.py
```

## Model Architecture
- Uses a deep learning model trained on hand gesture data
- Implements MediaPipe for accurate hand landmark detection
- Uses TensorFlow/Keras for the neural network implementation
- Includes pretrained models for both alphabets and phrases

## Training Data
The project includes two main datasets:
- Alphabet_Data: Contains training data for A-Z alphabets
- MP_Data: Contains training data for common phrases

## Performance
- Real-time detection with minimal latency
- High accuracy in controlled environments
- Robust to different lighting conditions
- Supports multiple hand orientations

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Improvements
- Support for more complex phrases
- Integration with text-to-speech
- Mobile application development
- Support for multiple simultaneous users
- Improved accuracy in varying light conditions

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Contact
- GitHub: [@dilip-ravichandra](https://github.com/dilip-ravichandra)

## Acknowledgments
- MediaPipe team for their hand tracking solution
- TensorFlow and Keras documentation
- OpenCV community for computer vision tools.