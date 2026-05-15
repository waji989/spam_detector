import customtkinter as ctk
from tkinter import filedialog
from model import load_models, predict

# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Load models
vectorizer, nb_model, lr_model, svm_model, metrics = load_models()

# App window
app = ctk.CTk()
app.title("📧 AI Spam Detection System")
app.geometry("1000x650")

# ================= HEADER =================
header = ctk.CTkFrame(app, height=60)
header.pack(fill="x")

title = ctk.CTkLabel(
    header,
    text="📧 Email Spam Detection Dashboard",
    font=("Arial", 22, "bold")
)
title.pack(pady=15)

# ================= MAIN =================
main = ctk.CTkFrame(app)
main.pack(fill="both", expand=True, padx=20, pady=20)

# ========== LEFT (EMAIL INPUT) ==========
left = ctk.CTkFrame(main)
left.pack(side="left", fill="both", expand=True, padx=10)

ctk.CTkLabel(left, text="✉️ Email Content", font=("Arial", 16)).pack(pady=10)

input_box = ctk.CTkTextbox(left, height=300)
input_box.pack(fill="both", expand=True, padx=10, pady=10)

def upload_file():
    file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file:
        with open(file, "r", encoding="utf-8") as f:
            input_box.delete("1.0", "end")
            input_box.insert("1.0", f.read())

ctk.CTkButton(left, text="📁 Upload Email (.txt)", command=upload_file).pack(pady=5)

# ========== RIGHT (RESULTS) ==========
right = ctk.CTkFrame(main)
right.pack(side="right", fill="both", expand=True, padx=10)

ctk.CTkLabel(right, text="📊 Analysis Result", font=("Arial", 16)).pack(pady=10)

result_label = ctk.CTkLabel(right, text="", font=("Arial", 14), justify="left")
result_label.pack(pady=10)

# Progress bars
ctk.CTkLabel(right, text="Naive Bayes").pack()
nb_bar = ctk.CTkProgressBar(right, width=300)
nb_bar.pack(pady=5)

ctk.CTkLabel(right, text="Logistic Regression").pack()
lr_bar = ctk.CTkProgressBar(right, width=300)
lr_bar.pack(pady=5)

# URL risk bar
ctk.CTkLabel(right, text="URL Risk").pack()
url_bar = ctk.CTkProgressBar(right, width=300)
url_bar.pack(pady=5)

# ========== ANALYZE FUNCTION ==========
def analyze():
    text = input_box.get("1.0", "end").strip()
    
    if not text:
        result_label.configure(text="⚠️ Please enter email content")
        return
    
    nb, lr, url_score, final = predict(
        text, vectorizer, nb_model, lr_model, svm_model
    )
    
    nb_bar.set(nb / 100)
    lr_bar.set(lr / 100)
    url_bar.set(url_score / 100)
    
    # Risk level
    avg = (nb + lr) / 2 + url_score
    
    if final:
        status = "🚨 SPAM DETECTED"
        color = "red"
    else:
        status = "✅ SAFE EMAIL"
        color = "green"
    
    result_label.configure(
        text=f"{status}\n\n"
             f"Naive Bayes: {nb:.2f}%\n"
             f"Logistic Regression: {lr:.2f}%\n"
             f"URL Risk: {url_score:.2f}%\n\n"
             f"Overall Risk Score: {avg:.2f}%"
    )

# ========== BUTTON ==========
ctk.CTkButton(
    app,
    text="🔍 Analyze Email",
    command=analyze,
    height=45,
    font=("Arial", 14, "bold")
).pack(pady=10)

# ================= FOOTER =================
footer = ctk.CTkLabel(
    app,
    text="AI Spam Detection System | Final Year Project",
    font=("Arial", 10)
)
footer.pack(pady=5)

app.mainloop()