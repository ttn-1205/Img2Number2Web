AI-Powered Digit Drawing & Web Submission Tool

This project is an interactive handwritten digit recognition app powered by a fine-tuned CNN based on the MNIST dataset. Users draw digits as answers to math questions on [Zetamac's arithmetic speed game](https://arithmetic.zetamac.com/), and the app automatically processes, predicts, and submits the answer in near real-time.

1. Features
- Draw digits using a Tkinter canvas
- Predicts digits using a fine-tuned MNIST CNN
- Automatically detects and isolates digits
- Submits predictions to Zetamac via Selenium
- Lightweight, intuitive UI with live feedback

2. Model Info
- `Initial_MNIST.keras`: Base model trained on raw MNIST
- `Finetuned_MNIST.keras`: Refined using personal digit samples for better accuracy

3. Dependencies
Install all dependencies via: pip install -r requirements.txt

4. How to Run
- Clone the repository and ensure all files are present:
'main.py'
'Finetuned_MNIST.keras'
'requirements.txt'
- Run the script: python main.py

A drawing canvas will appear. Draw your answer to the Zetamac question.
Double-click the canvas to predict and automatically submit your answer.

- You can watch the demo video included in the folder

