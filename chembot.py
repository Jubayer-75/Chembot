# chemistry_bot.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- base dataset: chemistry Q/A pairs ---
data = [
    ("what is water", "Water is H2O, natural 💧"),
    ("formula of water", "H2O, duh 😎"),
    ("what is hydrogen", "Hydrogen is H, lightest element 🌬️"),
    ("what is oxygen", "Oxygen is O, helps you breathe 😤"),
    ("atomic number of hydrogen", "1 🔢"),
    ("atomic number of oxygen", "8 🔢"),
    ("what is acid", "Acid = proton donor, strong taste 😝"),
    ("what is base", "Base = proton acceptor, slippery vibes 🧼"),
    ("what is salt", "Salt = acid + base chill 😎"),
    ("who is your creator?", "Jubayer"),
    ("sal", "Acid and base"),
]

# --- periodic table dataset (118 elements) ---
elements_data = [
    ("hydrogen", "Hydrogen (H), Atomic number 1, Mass ~1"),
    ("helium", "Helium (He), Atomic number 2, Mass ~4"),
    ("lithium", "Lithium (Li), Atomic number 3, Mass ~7"),
    ("beryllium", "Beryllium (Be), Atomic number 4, Mass ~9"),
    ("boron", "Boron (B), Atomic number 5, Mass ~11"),
    ("carbon", "Carbon (C), Atomic number 6, Mass ~12"),
    ("nitrogen", "Nitrogen (N), Atomic number 7, Mass ~14"),
    ("oxygen", "Oxygen (O), Atomic number 8, Mass ~16"),
    ("fluorine", "Fluorine (F), Atomic number 9, Mass ~19"),
    ("neon", "Neon (Ne), Atomic number 10, Mass ~20"),
    ("sodium", "Sodium (Na), Atomic number 11, Mass ~23"),
    ("magnesium", "Magnesium (Mg), Atomic number 12, Mass ~24"),
    ("aluminium", "Aluminium (Al), Atomic number 13, Mass ~27"),
    ("silicon", "Silicon (Si), Atomic number 14, Mass ~28"),
    ("phosphorus", "Phosphorus (P), Atomic number 15, Mass ~31"),
    ("sulfur", "Sulfur (S), Atomic number 16, Mass ~32"),
    ("chlorine", "Chlorine (Cl), Atomic number 17, Mass ~35.5"),
    ("argon", "Argon (Ar), Atomic number 18, Mass ~40"),
    ("potassium", "Potassium (K), Atomic number 19, Mass ~39"),
    ("calcium", "Calcium (Ca), Atomic number 20, Mass ~40"),
    ("scandium", "Scandium (Sc), Atomic number 21, Mass ~45"),
    ("titanium", "Titanium (Ti), Atomic number 22, Mass ~48"),
    ("vanadium", "Vanadium (V), Atomic number 23, Mass ~51"),
    ("chromium", "Chromium (Cr), Atomic number 24, Mass ~52"),
    ("manganese", "Manganese (Mn), Atomic number 25, Mass ~55"),
    ("iron", "Iron (Fe), Atomic number 26, Mass ~56"),
    ("cobalt", "Cobalt (Co), Atomic number 27, Mass ~59"),
    ("nickel", "Nickel (Ni), Atomic number 28, Mass ~59"),
    ("copper", "Copper (Cu), Atomic number 29, Mass ~64"),
    ("zinc", "Zinc (Zn), Atomic number 30, Mass ~65"),
    ("gallium", "Gallium (Ga), Atomic number 31, Mass ~70"),
    ("germanium", "Germanium (Ge), Atomic number 32, Mass ~73"),
    ("arsenic", "Arsenic (As), Atomic number 33, Mass ~75"),
    ("selenium", "Selenium (Se), Atomic number 34, Mass ~79"),
    ("bromine", "Bromine (Br), Atomic number 35, Mass ~80"),
    ("krypton", "Krypton (Kr), Atomic number 36, Mass ~84"),
    ("rubidium", "Rubidium (Rb), Atomic number 37, Mass ~85"),
    ("strontium", "Strontium (Sr), Atomic number 38, Mass ~88"),
    ("yttrium", "Yttrium (Y), Atomic number 39, Mass ~89"),
    ("zirconium", "Zirconium (Zr), Atomic number 40, Mass ~91"),
    ("niobium", "Niobium (Nb), Atomic number 41, Mass ~93"),
    ("molybdenum", "Molybdenum (Mo), Atomic number 42, Mass ~96"),
    ("technetium", "Technetium (Tc), Atomic number 43, Mass ~98"),
    ("ruthenium", "Ruthenium (Ru), Atomic number 44, Mass ~101"),
    ("rhodium", "Rhodium (Rh), Atomic number 45, Mass ~103"),
    ("palladium", "Palladium (Pd), Atomic number 46, Mass ~106"),
    ("silver", "Silver (Ag), Atomic number 47, Mass ~108"),
    ("cadmium", "Cadmium (Cd), Atomic number 48, Mass ~112"),
    ("indium", "Indium (In), Atomic number 49, Mass ~115"),
    ("tin", "Tin (Sn), Atomic number 50, Mass ~119"),
    ("antimony", "Antimony (Sb), Atomic number 51, Mass ~122"),
    ("tellurium", "Tellurium (Te), Atomic number 52, Mass ~128"),
    ("iodine", "Iodine (I), Atomic number 53, Mass ~127"),
    ("xenon", "Xenon (Xe), Atomic number 54, Mass ~131"),
    ("cesium", "Cesium (Cs), Atomic number 55, Mass ~133"),
    ("barium", "Barium (Ba), Atomic number 56, Mass ~137"),
    ("lanthanum", "Lanthanum (La), Atomic number 57, Mass ~139"),
    ("cerium", "Cerium (Ce), Atomic number 58, Mass ~140"),
    ("praseodymium", "Praseodymium (Pr), Atomic number 59, Mass ~141"),
    ("neodymium", "Neodymium (Nd), Atomic number 60, Mass ~144"),
    ("promethium", "Promethium (Pm), Atomic number 61, Mass ~145"),
    ("samarium", "Samarium (Sm), Atomic number 62, Mass ~150"),
    ("europium", "Europium (Eu), Atomic number 63, Mass ~152"),
    ("gadolinium", "Gadolinium (Gd), Atomic number 64, Mass ~157"),
    ("terbium", "Terbium (Tb), Atomic number 65, Mass ~159"),
    ("dysprosium", "Dysprosium (Dy), Atomic number 66, Mass ~162"),
    ("holmium", "Holmium (Ho), Atomic number 67, Mass ~165"),
    ("erbium", "Erbium (Er), Atomic number 68, Mass ~167"),
    ("thulium", "Thulium (Tm), Atomic number 69, Mass ~169"),
    ("ytterbium", "Ytterbium (Yb), Atomic number 70, Mass ~173"),
    ("lutetium", "Lutetium (Lu), Atomic number 71, Mass ~175"),
    ("hafnium", "Hafnium (Hf), Atomic number 72, Mass ~178"),
    ("tantalum", "Tantalum (Ta), Atomic number 73, Mass ~181"),
    ("tungsten", "Tungsten (W), Atomic number 74, Mass ~184"),
    ("rhenium", "Rhenium (Re), Atomic number 75, Mass ~186"),
    ("osmium", "Osmium (Os), Atomic number 76, Mass ~190"),
    ("iridium", "Iridium (Ir), Atomic number 77, Mass ~192"),
    ("platinum", "Platinum (Pt), Atomic number 78, Mass ~195"),
    ("gold", "Gold (Au), Atomic number 79, Mass ~197"),
    ("mercury", "Mercury (Hg), Atomic number 80, Mass ~201"),
    ("thallium", "Thallium (Tl), Atomic number 81, Mass ~204"),
    ("lead", "Lead (Pb), Atomic number 82, Mass ~207"),
    ("bismuth", "Bismuth (Bi), Atomic number 83, Mass ~209"),
    ("polonium", "Polonium (Po), Atomic number 84, Mass ~209"),
    ("astatine", "Astatine (At), Atomic number 85, Mass ~210"),
    ("radon", "Radon (Rn), Atomic number 86, Mass ~222"),
    ("francium", "Francium (Fr), Atomic number 87, Mass ~223"),
    ("radium", "Radium (Ra), Atomic number 88, Mass ~226"),
    ("actinium", "Actinium (Ac), Atomic number 89, Mass ~227"),
    ("thorium", "Thorium (Th), Atomic number 90, Mass ~232"),
    ("protactinium", "Protactinium (Pa), Atomic number 91, Mass ~231"),
    ("uranium", "Uranium (U), Atomic number 92, Mass ~238"),
    ("neptunium", "Neptunium (Np), Atomic number 93, Mass ~237"),
    ("plutonium", "Plutonium (Pu), Atomic number 94, Mass ~244"),
    ("americium", "Americium (Am), Atomic number 95, Mass ~243"),
    ("curium", "Curium (Cm), Atomic number 96, Mass ~247"),
    ("berkelium", "Berkelium (Bk), Atomic number 97, Mass ~247"),
    ("californium", "Californium (Cf), Atomic number 98, Mass ~251"),
    ("einsteinium", "Einsteinium (Es), Atomic number 99, Mass ~252"),
    ("fermium", "Fermium (Fm), Atomic number 100, Mass ~257"),
    ("mendelevium", "Mendelevium (Md), Atomic number 101, Mass ~258"),
    ("nobelium", "Nobelium (No), Atomic number 102, Mass ~259"),
    ("lawrencium", "Lawrencium (Lr), Atomic number 103, Mass ~266"),
    ("rutherfordium", "Rutherfordium (Rf), Atomic number 104, Mass ~267"),
    ("dubnium", "Dubnium (Db), Atomic number 105, Mass ~268"),
    ("seaborgium", "Seaborgium (Sg), Atomic number 106, Mass ~269"),
    ("bohrium", "Bohrium (Bh), Atomic number 107, Mass ~270"),
    ("hassium", "Hassium (Hs), Atomic number 108, Mass ~277"),
    ("meitnerium", "Meitnerium (Mt), Atomic number 109, Mass ~278"),
    ("darmstadtium", "Darmstadtium (Ds), Atomic number 110, Mass ~281"),
    ("roentgenium", "Roentgenium (Rg), Atomic number 111, Mass ~282"),
    ("copernicium", "Copernicium (Cn), Atomic number 112, Mass ~285"),
    ("nihonium", "Nihonium (Nh), Atomic number 113, Mass ~286"),
    ("flerovium", "Flerovium (Fl), Atomic number 114, Mass ~289"),
    ("moscovium", "Moscovium (Mc), Atomic number 115, Mass ~290"),
    ("livermorium", "Livermorium (Lv), Atomic number 116, Mass ~293"),
    ("tennessine", "Tennessine (Ts), Atomic number 117, Mass ~294"),
    ("oganesson", "Oganesson (Og), Atomic number 118, Mass ~294"),
]

# --- merge everything into training data🥱 ---
data.extend(elements_data)

# --- build vocabulary from questions only (keys) ---
words = list(set(" ".join([d[0] for d in data]).lower().split()))
word2idx = {w: i for i, w in enumerate(words)}

# --- labels: preserve first-seen order but unique📌 ---
# using dict.fromkeys keeps the first occurrence order (py3.7+)
labels = list(dict.fromkeys([d[1] for d in data]))
label2idx = {l: i for i, l in enumerate(labels)}

# --- encode sentences to one-hot vectors ---
def encode_sentence(sentence):
    vec = np.zeros(len(words), dtype=np.float32)
    for w in sentence.lower().split():
        if w in word2idx:
            vec[word2idx[w]] = 1.0
    return vec

X = np.array([encode_sentence(d[0]) for d in data], dtype=np.float32)
y = np.array([label2idx[d[1]] for d in data], dtype=np.int64)

# --- convert to torch tensors ---
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# --- double hidden layer neural network ---
class ChemBotNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model size scaled for many labels
model = ChemBotNet(len(words), 128, 64, len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- training ---
print("Training started... this may take a little while depending on your CPU/GPU.")
for epoch in range(600):
    model.train()
    out = model(X_tensor)
    loss = criterion(out, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.4f}")

# --- chatting function ---
def chat(sentence):
    model.eval()
    vec = torch.from_numpy(encode_sentence(sentence)).float()
    with torch.no_grad():
        out = model(vec)
        pred = torch.argmax(out).item()
    return labels[pred]

# --- interactive loop ---
print("ChemBot(by_Jubayer) ready with full 118 elements. Type 'quit' to exit 😎")
while True:
    msg = input("You: ").strip()
    if not msg:
        continue
    if msg.lower() == "quit":
        print("Peace out ✌️")
        break
    reply = chat(msg)
    print("ChemBot:", reply)
