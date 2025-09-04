---

# Shinigami Chess Engine (V.1.18.2 – Gen 2 Edition)

![MIT License](https://img.shields.io/badge/license-MIT-FF4136?labelColor=gray)
![Python](https://img.shields.io/badge/language-Python_3.8+-2ECC40?labelColor=gray)
![Creator](https://img.shields.io/badge/Creator_Name-Tonmoy_KS-0074D9?labelColor=gray)

**Shinigami V.1.18.2 – Gen 2 Edition (Latest)**  
_A professional chess engine with full tree parallelization, advanced NNUE, LLM-powered move explanations, self-adapting features, and a brutal/funny trash talk system._

---

## 🚀 Latest Updates (V.1.18.2)
- **Genetic Feature Engineering:** DEAP-powered auto-tuning for piece values and PSTs mid-game.
- **Dynamic Opening Book Pruning:** Opening book now self-prunes weak lines and adapts to new opponents.
- **Enhanced Evaluation:**  
  - King safety, pawn structure, mobility, outpost, bishop pair, rook/queen evaluation refined.
- **Full UCI Protocol:** Full UCI compliance.
- **Logging:** Cleaner, more informative logs for games, search, and training.
- **Opponent Learning:** Engine adapts to your style in real time.
- **Debug:** Fixed a Lot of Bugs and made the code more cleaner and organized.

---

## Features

- **Full Tree Parallelization:** Multicore search for epic depth and speed.
- **Advanced NNUE (HalfKAv2):** Modern neural evaluation (user-trainable).
- **CNN Policy Network:** Move ordering powered by a convolutional neural net.
- **Syzygy Tablebase Support:** Endgame perfection (≤7 pieces).
- **Genetic Feature Engineering:** DEAP-powered auto-tuning for piece values and tables.
- **Self-Play & Training:** Generates and learns from games, retrains NNUE and policy networks.
- **Dynamic Opening Book:** Adapts with self-play and opponent moves, prunes weak lines.
- **Opponent Learning:** Adapts to your style, learning openings and move qualities.
- **Puzzle Generator:** Built-in tactical puzzles and board tasks.
- **GUI & Console:** Tkinter GUI or classic terminal mode.
- **UCI Protocol:** Plug into chess GUIs (Arena, CuteChess, etc.).
- **Trash Talk Engine:** Witty, customizable, optionally brutal commentary.
- **LLM Explanations:** In-character move explanations via OpenAI API (optional).
- **Extreme Difficulty Modes:** From beginner to "The Big Bang" (with cosmic warnings...).

---

## Installation

**Clone the Repository:**
```bash
git clone https://github.com/Tonmoy-KS/Shinigami-Chess-Engine
cd Shinigami-Chess-Engine
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```
*Requires Python 3.8+.*

### Optional Assets
- **NNUE Weights:** Download `nnue_weights.bin` (HalfKAv2 compatible) and place in project root or specify with `--nnue-file`.
- **Syzygy Tablebases:** Download and place .rtb files in `./tablebases` or specify via `--syzygy-path`.
- **Polyglot Opening Book:** Place `.bin` file in project root or specify via command-line.

---

## Usage

**Console Mode:**
```bash
python3 Main_Code_1
```

**GUI Mode:**
```bash
python3 Main_Code_1 --gui
```

**Self-Play & Training:**
```bash
python3 Main_Code_1 --self-play 100
```

**Custom Paths & Cores:**
```bash
python3 Main_Code_1 --cores 4 --nnue-file /path/to/my_nnue.bin --syzygy-path /path/to/my_tablebases
```

**UCI Protocol (for chess GUIs):**
```bash
python3 Main_Code_1 --uci
```

---

## Logging

Game statistics, self-play, and learning data are logged to `shinigami_engine.log`.

---

## LLM-Powered Move Explanations

- Uses OpenAI API (set `OPENAI_API_KEY` environment variable).
- Optional, can be enabled/disabled at runtime.
- Provides witty, in-character commentary on engine moves.

---

## Extreme Difficulty Modes

### Official
- **easy:** Beginner
- **medium:** Club-level
- **hard:** Strong player
- **god-of-death:** near Grandmaster
- **puzzle:** Tactical puzzle mode

### Experimental/Jokes
- **masochist:** Insane mode
- **dialing-satan-s-number:** Meme mode
- **the-big-bang:** God mode

---

## FAQ

1. **What makes Shinigami different from other chess engines?**  
  A: Genetic feature engineering, trash talk, and LLM-powered move explanations made it a great UX software and different from other Great engines that are just Silent Calculators.

2. **How do I enable LLM Explanations?**  
  A: Set your OpenAI API key and OpenAI model and enable at startup,we also have a LLM model but it's currently not available.

3. **How do I enable/disable The Trash Talk Dictionary?**  
  A: Enabled by default. You can customize or disable in the source code.

4. **What are “The Big Bang” and “Dialing Satan’s Number” modes?**  
  A: Extreme/joke modes. “The Big Bang” is blocked for safety; “Dialing Satan’s Number” is a meme. just to test the limits of Computation.

5. **Does Shinigami learn/adapt openings from games?**  
  A: Yes, via dynamic pruning and opponent learning.

6. **How do I train NNUE and Policy Network?**  
  A: Use self-play, or use your own datasets via Training a PyTorch model. so official Model currently.

7. **Is Shinigami UCI-compliant for GUIs?**  
  A: Yes.

8. **Can I contribute neural weights, feature sets, or puzzles?**  
  A: contributions welcome.

9. **What about system requirements for extreme modes?**  
  A: Use first five modes are Official modes that are Stable; the other modes are just experiments.

10. **Can I customize the playing style (aggressive, positional)?**  
   A: Not currently. planned for future updates.

---

## License

**License:** MIT  
**Credit:** Tonmoy-KS  
Do not claim as your own; fork, mod, and contribute instead!

---

## Major Contributers

1) [Inskitz](https://github.com/InSkitz)
[https://github.com/InSkitz]
2) [Dakter-Tech](https://github.com/dakther-tech) 
[https://github.com/dakther-tech]
3) [Shanjid-KS](https://github.com/shanjid-KS) 
[https://github.com/shanjid-KS]

---

#### How to Contribute
Pull requests, feature ideas, and code reviews are welcome! For major contributions, open an issue first.

---

## Contact

GitHub: [Tonmoy-KS](https://github.com/Tonmoy-KS)

---

*Happy reaping in 64 squares!* — bye

---