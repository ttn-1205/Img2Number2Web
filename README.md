### AI-Powered Digit Drawing & Web Submission Tool

This project is an interactive handwritten digit recognition app powered by a fine-tuned CNN based on the MNIST dataset. Users draw digits as answers to math questions on [Zetamac's arithmetic speed game](https://arithmetic.zetamac.com/), and the app automatically processes, predicts, and submits the answer in near real-time.

### 1. Features
- Draw digits using a Tkinter canvas
- Predicts digits using a fine-tuned MNIST CNN
- Automatically detects and isolates digits
- Submits predictions to Zetamac via Selenium
- Tkinter interactive UI with live notification after answer submission

### 2. Model Info
- `Initial_MNIST.keras`: Base model trained on raw MNIST.
- `Finetuned_MNIST.keras`: Fine-tuned using personal digit samples for better accuracy.

### 3. How to Use
- Running the `main.py` file
- A drawing canvas will appear. Draw your answer to the Zetamac question.
Double-click the canvas to predict and automatically submit your answer.
- You can watch the demo video included in the folder.

### 4. Dependencies
Install all dependencies via:
```bash
pip install -r requirements.txt
