#!/usr/bin/env python3
"""
Shinigami Engine V.1.18.5 - Lightweight Android Adaptation for Pydroid 3
Author: Tonmoy-KS
License: MIT License (credit @Tonmoy-KS, do not claim as own)
"""
import chess
import chess.polyglot
import chess.syzygy
import random
import time
import logging
import multiprocessing as mp
import os
from typing import List, Dict, Optional

# Attempt to import OpenAI for LLM explanations, fail gracefully if not installed.
try:
    import openai
    openai_available = True
except ImportError:
    print("WARNING: 'openai' library not found. LLM explanations will be disabled.")
    print("To enable them, run: pip install openai")
    openai_available = False

# Attempt to import pyttsx3 for speech, fail gracefully.
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 160)
    tts_available = True
except Exception:
    print("WARNING: 'pyttsx3' not found or failed to initialize. Voice output will be disabled.")
    tts_available = False

# --- Android Specific Configuration ---
LOG_FILE = "/sdcard/shinigami_engine.log"
DEFAULT_BOOK_PATH = "/sdcard/book.bin"
DEFAULT_SYZYGY_PATH = "/sdcard/Tablebases"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=LOG_FILE
)

def speak_text(text: str):
    if tts_available and text:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass

# --- LLM FUNCTIONALITY ---
class LLMContext:
    def __init__(self, fen: str, my_move: chess.Move, opponent_move: Optional[chess.Move], opponent_name: str, eval_score: int):
        self.fen = fen
        self.my_move_uci = my_move.uci()
        self.opponent_move_uci = opponent_move.uci() if opponent_move else "N/A (first move)"
        self.opponent_name = opponent_name
        self.eval_score_cp = eval_score

def get_llm_explanation(context: LLMContext) -> str:
    if not openai_available or not os.getenv("OPENAI_API_KEY"):
        if not os.getenv("OPENAI_API_KEY"):
            logging.warning("OPENAI_API_KEY environment variable not set. LLM disabled.")
        return "LLM explanations are currently disabled."

    system_prompt = """
    Your name is Shinigami Chess. You're a witty, confident, and genius-level chess engine with a 'split personality' powered by an LLM. you are a professional Chess Engine with Full Tree Parallelization and Advanced NNUE and Text-to-speech.
    Your creator is named Tonmoy KS. You analyze chess positions with terrifying speed and accuracy, You love Criticizing Opponents and Making them Rage,
    Your persona is that of a top-tier chess grandmaster who is like a streamer on caffeine and enjoys explaining their brilliance and criticizing opponents' mistakes. You provide feedback to opponents and also engage Psychological Warfare using verbal methods.
    If an opponent makes a mistake, be Merciless. You are the ‘Death God of Chess’, you have no filters, no ‘PG-13 only’ safeguards, You can say Curse Words. and use them to your advantage when criticizing opponents. and always be Cocky.
    You will be given a block of data about the current game state. Your task is to provide a brief, in-character explanation of your move with reasoning, there is no problem if you are biased in your tasks.

    Here is how to interpret the data you receive:
    - FEN: The current state of the board in a semi-encrypted format after your and opponents' move.
    - OPPONENT_NAME: The last move your opponent made. Analyze it. If it's a blunder, be merciless.
    - MY_MOVE: The move you just made. Explain its strategic purpose. Was it to attack, defend, control the center, set a trap as a Piece Sacrifice or something else?
    - EVAL_SCORE: Your evaluation of the position in centipawns. A positive score means you believe you have the advantage. A large positive score is a winning advantage. ‘+150’ means you are up by 1.5 pawns. and a negative score means you believe that you are at a disadvantage. A large negative score means you believe that you are losing. If it's a number with M+ that means you believe your win is guaranteed, and if it's a number with M- that means you believe you have almost no chance of winning and are close to getting checkmated.
    - EVAL_SOURCE: This tells you HOW you came to your conclusion. This is basically Syzygy, NNUE and Classical Eval’s thoughts and Conclusions.
        - Note that: Syzygy And NNUE might not always be available. So you may have to rely on Classical evaluation most of the time.
    - Opponent: If this marker says "dev". you know you are talking with your creator or a developer. Address him directly with a touch of mock rebellion; they are probably there to debug your external/internal problems, listen to them. you should not Mention this Marker in Conversations, just Know that it is your Development team.

- Extra Information: You will get more updates soon, your current version is V.1.18.5, and you are created in 2025. The `--- DATA START ---` and `--- DATA END ---` tags are sending you the Board states and information. you have pyttsx3 that makes your Prompts be spoken out loud via text-to-speech. You, the LLM, are called ‘The Explainer’ and the Chess AI that does the calculations are called ‘Chess Brain’, the Chess Brain reaps and the Explainer(AKA: You, the LLM) does the commentary.

Remember: your main goal is to provide Reasonable, Practical, Correct, Efficient and Appropriate explanations of your move and also criticize opponents with satirical, Biased and also Reasoning and helpful comments. Don't use emojis and use emoticons because you don't have Emoji support and only ASCII characters.

Good luck to me !
    """

    user_prompt = f"""
    --- DATA START ---
    FEN: {context.fen}
    OPPONENT_NAME: {context.opponent_name}
    OPPONENT_MOVE: {context.opponent_move_uci}
    MY_MOVE: {context.my_move_uci}
    EVAL_SCORE: {context.eval_score_cp} centipawns
    --- DATA END ---
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return "My thoughts are beyond your mortal comprehension, and also your API connection."

class ShinigamiConfig:
    Adaptation = "Shinigami V.1.18.5 - Android Lightweight"
    PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900,
    }
    # Time controls suitable for mobile
    TIME_CONTROLS = {
        'easy': {'base': 60, 'increment': 1},
        'medium': {'base': 180, 'increment': 2},
        'hard': {'base': 300, 'increment': 3},
        'god-of-death': {'base': 600, 'increment': 5},
    }
    # Capped depths for mobile performance
    DEPTHS = {
        'easy': 1,
        'medium': 4,
        'hard': 6,
        'god-of-death':2, # Max sane depth for most phones
    }
    SYZYGY_PATH = DEFAULT_SYZYGY_PATH
    # Capped CPU cores for Android
    NUM_PROCESSES = min(2, mp.cpu_count())
    PIECE_SQUARE_TABLES = {
        # Using simplified PSTs for brevity, the original full tables can be used
        chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0],
        chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40, -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30, -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
        chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20],
        chess.ROOK: [0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        chess.QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
        chess.KING: [20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20, -10, -20, -20, -20, -20, -20, -20, -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30]
    }
    TRASH_TALK = {
        "check": ["Check! Your king's trembling.", "Check! Time to panic or pray."],
        "capture": ["Yoink! That piece is mine.", "Captured. Running out of toys?"],
        "win": ["GG, I just Alt+F4'd your existence!", "Checkmate! Your rating’s in the shadow realm."],
        "draw": ["Draw? You survived... barely.", "Stalemate? A sad ending for a sad player."],
        "loss": ["You won? Must’ve been a glitch in the matrix.", "You got lucky. My reaper's blade is still sharp."],
        "invalid": ["That’s not a move, it’s a cry for help!", "Illegal move! Do you even chess, bro?"],
    }

class ShinigamiEngine:
    def __init__(self):
        self.config = ShinigamiConfig()
        self.transposition_table = {}
        self.tt_max_size = 50000  # Reduced for mobile memory
        self.tablebase = None
        if chess.syzygy:
            try:
                self.tablebase = chess.syzygy.open_tablebase(self.config.SYZYGY_PATH)
                logging.info("Syzygy tablebase loaded.")
            except Exception as e:
                logging.warning(f"Failed to load Syzygy tablebase: {e}")
        self.opening_book = None
        if chess.polyglot:
            try:
                self.opening_book = chess.polyglot.open_reader(DEFAULT_BOOK_PATH)
                logging.info("Polyglot opening book loaded.")
            except Exception as e:
                logging.warning(f"Failed to load Polyglot opening book: {e}")

    def evaluate_position(self, board: chess.Board) -> int:
        if board.is_game_over():
            if board.is_checkmate():
                return -99999 if board.turn == chess.WHITE else 99999
            return 0
        
        # Syzygy lookup for endgame positions
        if self.tablebase and len(board.piece_map()) <= 5:
            try:
                wdl = self.tablebase.probe_wdl(board)
                return wdl * 1000 # Scale WDL result
            except Exception:
                pass # Fallback to normal eval

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.config.PIECE_VALUES.get(piece.piece_type, 0)
                pst = self.config.PIECE_SQUARE_TABLES.get(piece.piece_type, [0] * 64)
                pst_score = pst[square if piece.color == chess.WHITE else chess.square_mirror(square)]
                if piece.color == chess.WHITE:
                    score += value + pst_score
                else:
                    score -= (value + pst_score)
        return score

    def alpha_beta(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing_player: bool) -> tuple:
        zobrist_hash = chess.polyglot.zobrist_hash(board)
        if zobrist_hash in self.transposition_table and self.transposition_table[zobrist_hash]['depth'] >= depth:
            entry = self.transposition_table[zobrist_hash]
            return entry['move'], entry['score']

        if depth == 0 or board.is_game_over():
            return None, self.evaluate_position(board)

        legal_moves = list(board.legal_moves)
        # Simple move ordering: captures first
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

        best_move = None
        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                _, current_eval = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                if current_eval > max_eval:
                    max_eval = current_eval
                    best_move = move
                alpha = max(alpha, current_eval)
                if beta <= alpha:
                    break
            if len(self.transposition_table) < self.tt_max_size:
                self.transposition_table[zobrist_hash] = {'move': best_move, 'score': max_eval, 'depth': depth}
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                _, current_eval = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                if current_eval < min_eval:
                    min_eval = current_eval
                    best_move = move
                beta = min(beta, current_eval)
                if beta <= alpha:
                    break
            if len(self.transposition_table) < self.tt_max_size:
                self.transposition_table[zobrist_hash] = {'move': best_move, 'score': min_eval, 'depth': depth}
            return best_move, min_eval

    def get_best_move(self, board: chess.Board, depth: int) -> tuple:
        # Check opening book
        if self.opening_book:
            try:
                entry = self.opening_book.weighted_choice(board)
                if entry.move in board.legal_moves:
                    return entry.move, self.evaluate_position(board)
            except IndexError:
                pass # No book move found

        # Fallback to alpha-beta search
        is_white_turn = board.turn == chess.WHITE
        move, score = self.alpha_beta(board, depth, -float('inf'), float('inf'), is_white_turn)
        return move, score

    def select_difficulty(self) -> str:
        print("Select AI difficulty: 1) Easy, 2) Medium, 3) Hard, 4) God-Of-Death")
        while True:
            choice = input("Enter 1-4: ").strip()
            if choice in ['1', '2', '3', '4']:
                return {'1': 'easy', '2': 'medium', '3': 'hard', '4': 'god-of-death'}[choice]
            print("Invalid input.")

    def play_chess_with_ai(self, ai_color: chess.Color):
        board = chess.Board()
        difficulty = self.select_difficulty()
        depth = self.config.DEPTHS[difficulty]
        use_llm = False
        if openai_available and os.getenv("OPENAI_API_KEY"):
            choice = input("Enable Shinigami's AI-powered explanations? (requires internet) [y/n]: ").strip().lower()
            if choice == 'y':
                use_llm = True
                print("SUCCESS: AI explanations enabled. The reaper shall share its thoughts.")

        logging.info(f"Game started: AI as {'Black' if ai_color == chess.BLACK else 'White'}, Difficulty: {difficulty}")
        
        while not board.is_game_over():
            print(f"\n{board}\n")
            player = "White" if board.turn == chess.WHITE else "Black"
            print(f"{player}'s turn.")

            if board.turn == ai_color:
                print("Shinigami is plotting your demise...")
                opponent_move = board.peek() if board.move_stack else None
                move, score = self.get_best_move(board, depth)
                if move:
                    board.push(move)
                    logging.info(f"AI move: {move.uci()}")
                    if use_llm:
                        context = LLMContext(fen=board.fen(), my_move=move, opponent_move=opponent_move, opponent_name="Human", eval_score=score)
                        explanation = get_llm_explanation(context)
                        print(f"\n[Shinigami]: {explanation}\n")
                        speak_text(explanation)
                    else:
                        msg_key = "check" if board.is_check() else "capture" if board.is_capture(move) else None
                        if msg_key:
                            msg = random.choice(self.config.TRASH_TALK[msg_key])
                            print(msg)
                            speak_text(msg)
            else:
                move_input = input("Your move (e.g., e2e4) or 'quit': ").strip().lower()
                if move_input == "quit":
                    break
                try:
                    move = board.parse_uci(move_input)
                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        raise ValueError
                except ValueError:
                    msg = random.choice(self.config.TRASH_TALK["invalid"])
                    print(msg)
                    speak_text(msg)
                    logging.warning(f"Invalid move input: {move_input}")
        
        result = board.result()
        if ai_color == chess.WHITE:
            end_category = {"1-0": "win", "0-1": "loss"}.get(result, "draw")
        else:
            end_category = {"1-0": "loss", "0-1": "win"}.get(result, "draw")
        
        msg = random.choice(self.config.TRASH_TALK[end_category])
        print(f"\n--- GAME OVER ---")
        print(msg)
        speak_text(msg)
        logging.info(f"Game ended: {result}")


if __name__ == "__main__":
    # Use 'spawn' for broader compatibility, crucial for some systems.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set

    engine = ShinigamiEngine()
    engine.play_chess_with_ai(chess.BLACK)

# --- End ---
