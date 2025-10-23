import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import os

class DarkStyle(ttk.Style):
    def __init__(self, root):
        super().__init__(root)
        self.theme_use('clam')
        self.configure('.', background='#1c1c2e', foreground='white', font=('Arial', 10))
        self.configure('Treeview', background='#2e2e44', foreground='white', fieldbackground='#2e2e44')
        self.configure('Treeview.Heading', background='#0d1a3a', foreground='white', font=('Arial', 10, 'bold'))
        self.map('TButton',
                 foreground=[('active', 'white')],
                 background=[('active', '#0055ff')])

class CatDogClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Breed Classifier")
        self.root.geometry("1050x700")
        self.root.configure(bg="#1c1c2e")
        DarkStyle(self.root)

        self.model = None
        self.class_labels = {}
        self.setup_model()
        self.image_paths = []

        self.setup_gui()

    def setup_model(self):
        try:
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.load_labels()
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def load_labels(self):
        # Full cat and dog breeds from ImageNet
        self.class_labels = {
            # Cats
            281: 'Tabby cat', 282: 'Tiger cat', 283: 'Persian cat', 284: 'Siamese cat', 285: 'Egyptian cat',
            # Dogs
            151: 'Chihuahua', 152: 'Japanese spaniel', 153: 'Maltese dog', 154: 'Pekinese', 155: 'Shih-Tzu',
            156: 'Blenheim spaniel', 157: 'Papillon', 158: 'Toy terrier', 159: 'Rhodesian ridgeback', 160: 'Afghan hound',
            161: 'Basset', 162: 'Beagle', 163: 'Bloodhound', 164: 'Bluetick', 165: 'Black-and-tan coonhound',
            166: 'Walker hound', 167: 'English foxhound', 168: 'Redbone', 169: 'Borzoi', 170: 'Irish wolfhound',
            171: 'Italian greyhound', 172: 'Whippet', 173: 'Ibizan hound', 174: 'Norwegian elkhound', 175: 'Otterhound',
            176: 'Saluki', 177: 'Scottish deerhound', 178: 'Weimaraner', 179: 'Staffordshire bullterrier',
            180: 'American Staffordshire terrier', 181: 'Bedlington terrier', 182: 'Border terrier', 183: 'Kerry blue terrier',
            184: 'Irish terrier', 185: 'Norfolk terrier', 186: 'Norwich terrier', 187: 'Yorkshire terrier', 188: 'Wire-haired fox terrier',
            189: 'Lakeland terrier', 190: 'Sealyham terrier', 191: 'Airedale', 192: 'Cairn', 193: 'Australian terrier',
            194: 'Dandie Dinmont', 195: 'Boston bull', 196: 'Miniature schnauzer', 197: 'Giant schnauzer', 198: 'Standard schnauzer',
            199: 'Scotch terrier', 200: 'Tibetan terrier', 201: 'Silky terrier', 202: 'Soft-coated wheaten terrier', 203: 'West Highland white terrier',
            204: 'Lhasa', 205: 'Flat-coated retriever', 206: 'Curly-coated retriever', 207: 'Golden retriever', 208: 'Labrador retriever',
            209: 'Chesapeake Bay retriever', 210: 'German short-haired pointer', 211: 'Vizsla', 212: 'English setter', 213: 'Irish setter',
            214: 'Gordon setter', 215: 'Brittany spaniel', 216: 'Clumber', 217: 'English springer', 218: 'Welsh springer spaniel',
            219: 'Cocker spaniel', 220: 'Sussex spaniel', 221: 'Irish water spaniel', 222: 'Kuvasz', 223: 'Schipperke', 224: 'Groenendael',
            225: 'Malinois', 226: 'Briard', 227: 'Kelpie', 228: 'Komondor', 229: 'Old English sheepdog', 230: 'Shetland sheepdog',
            231: 'Collie', 232: 'Border collie', 233: 'Bouvier des Flandres', 234: 'Rottweiler', 235: 'German shepherd', 236: 'Doberman',
            237: 'Miniature pinscher', 238: 'Greater Swiss Mountain dog', 239: 'Bernese mountain dog', 240: 'Appenzeller', 241: 'EntleBucher',
            242: 'Boxer', 243: 'Bull mastiff', 244: 'Tibetan mastiff', 245: 'French bulldog', 246: 'Great Dane', 247: 'Saint Bernard',
            248: 'Eskimo dog', 249: 'Malamute', 250: 'Siberian husky', 251: 'Dalmatian', 252: 'Affenpinscher', 253: 'Basenji', 254: 'Pug',
            255: 'Leonberg', 256: 'Newfoundland', 257: 'Great Pyrenees', 258: 'Samoyed', 259: 'Pomeranian', 260: 'Chow', 261: 'Keeshond',
            262: 'Brabancon griffon', 263: 'Pembroke', 264: 'Cardigan', 265: 'Toy poodle', 266: 'Miniature poodle', 267: 'Standard poodle',
            268: 'Mexican hairless', 269: 'Dingo', 270: 'Dhole', 271: 'African hunting dog'
        }

    def setup_gui(self):
        tk.Label(self.root, text="Cat vs Dog Breed Classifier",
                 font=('Arial', 18, 'bold'), bg="#0d1a3a", fg="white").pack(fill=tk.X, pady=(0,10))

        main_frame = tk.Frame(self.root, bg="#1c1c2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Left Panel ---
        left = tk.Frame(main_frame, bg="#1c1c2e")
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Selected Images:").pack(anchor='w', pady=(0,5))
        self.image_list = tk.Listbox(left, width=35, height=25, bg="#2e2e44", fg="white")
        self.image_list.pack(fill=tk.Y, expand=False)

        ttk.Button(left, text="Add Images", command=self.add_images).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Clear All", command=self.clear_images).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Analyze Images", command=self.analyze_images).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Clear Results", command=self.clear_results).pack(fill=tk.X, pady=3)

        # --- Right Panel ---
        right = tk.Frame(main_frame, bg="#1c1c2e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Results:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0,5))

        columns = ('Image', 'Animal Type', 'Breed', 'Confidence')
        self.results_tree = ttk.Treeview(right, columns=columns, show='headings', height=18)
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150, anchor='center')
        self.results_tree.pack(fill=tk.BOTH, expand=True, pady=(0,10))

        self.results_tree.bind("<<TreeviewSelect>>", self.show_selected_image)

        self.image_label = tk.Label(right, text="Image preview will appear here", bg="#1c1c2e", fg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=5)

        self.status_bar = tk.Label(self.root, text="Ready", bg="#0d1a3a", fg="white", anchor='w')
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def add_images(self):
        files = filedialog.askopenfilenames(title="Select images",
                                            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        for file in files:
            if file not in self.image_paths:
                self.image_paths.append(file)
                self.image_list.insert(tk.END, os.path.basename(file))

    def clear_images(self):
        self.image_paths.clear()
        self.image_list.delete(0, tk.END)
        self.clear_results()
        self.image_label.configure(image='', text="Image preview will appear here")
        self.status_bar.config(text="Ready")

    def clear_results(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def classify_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            class_id = top_catid[0].item()
            class_name = self.class_labels.get(class_id, f"class_{class_id}")
            confidence = top_prob[0].item()
            return class_name, confidence, image
        except Exception as e:
            print(e)
            return None, None, None

    def analyze_images(self):
        if not self.image_paths:
            messagebox.showwarning("Warning", "Please add images first!")
            return

        self.clear_results()

        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            predicted_class, confidence, image = self.classify_image(img_path)

            if not predicted_class:
                continue

            # Determine animal type
            if 'cat' in predicted_class.lower():
                animal_type = "Cat"
            elif 'dog' in predicted_class.lower() or 'hound' in predicted_class.lower():
                animal_type = "Dog"
            else:
                animal_type = "Dog"  # fallback, all breeds now covered

            # Clean breed name
            breed_name = predicted_class.replace('dog', '').replace('cat', '').strip().title()

            self.results_tree.insert('', tk.END, values=(
                filename, animal_type, breed_name, f"{confidence:.2%}"
            ))

        self.status_bar.config(text=f"Analysis complete for {len(self.image_paths)} images.")

    def show_selected_image(self, event):
        selected = self.results_tree.selection()
        if not selected:
            return
        item = selected[0]
        values = self.results_tree.item(item, 'values')
        filename = values[0]
        img_path = next((p for p in self.image_paths if os.path.basename(p) == filename), None)
        if img_path:
            img = Image.open(img_path)
            img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk


def main():
    root = tk.Tk()
    CatDogClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
