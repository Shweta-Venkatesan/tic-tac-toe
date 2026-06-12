// =============================================================================
// TicTacToeMaster.jsx — Complete AI-powered Tic-Tac-Toe (single file)
// AI Concepts: Minimax, Alpha-Beta Pruning, Game Theory, State Space Search
// Author: AI Internship Portfolio Project
// =============================================================================

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Zap, Brain, BookOpen, BarChart3, Cpu, GitBranch,
  RefreshCw, RotateCcw, Play, Square, Trophy, Handshake,
  User, Bot, ChevronDown, ChevronUp, Sparkles, Target,
  Activity, TrendingUp, Clock, Hash, Code2
} from 'lucide-react';

// ===== CONSTANTS =====

/** All 8 winning line combinations on a 3×3 board */
const WINNING_COMBINATIONS = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],   // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8],   // columns
  [0, 4, 8], [2, 4, 6]               // diagonals
];

/** Difficulty level configurations */
const DIFFICULTY_LEVELS = {
  EASY:   { label: 'Easy',   depth: 0, useAlphaBeta: false, description: 'Random moves — completely beatable' },
  MEDIUM: { label: 'Medium', depth: 3, useAlphaBeta: false, description: 'Limited 3-ply lookahead Minimax' },
  HARD:   { label: 'Hard',   depth: 9, useAlphaBeta: false, description: 'Full Minimax — no pruning' },
  EXPERT: { label: 'Expert', depth: 9, useAlphaBeta: true,  description: 'Minimax + Alpha-Beta Pruning' }
};

/** Game mode identifiers */
const GAME_MODES = {
  HVA: 'Human vs AI',
  HVH: 'Human vs Human',
  AVA: 'AI vs AI'
};

/** Timing delays in milliseconds */
const DELAY_MS = {
  AI_THINK: 900,
  AVA_MOVE: 1300
};

/** Theoretical maximum states in Tic-Tac-Toe (9 factorial) */
const MAX_STATES = 362880;

// ===== AI ENGINE =====

/**
 * Evaluates the current board state and returns a terminal score.
 * @param {Array<string|null>} board - 9-element board array
 * @param {string} aiSymbol - The AI's symbol ('X' or 'O')
 * @param {string} humanSymbol - The human's symbol ('X' or 'O')
 * @returns {number} +10 for AI win, -10 for human win, 0 otherwise
 */
function evaluateBoard(board, aiSymbol, humanSymbol) {
  for (const combo of WINNING_COMBINATIONS) {
    const [a, b, c] = combo;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      if (board[a] === aiSymbol) return 10;
      if (board[a] === humanSymbol) return -10;
    }
  }
  return 0;
}

/**
 * Returns all indices on the board that are currently empty (null).
 * @param {Array<string|null>} board - 9-element board array
 * @returns {number[]} Array of available cell indices
 */
function getAvailableMoves(board) {
  return board.reduce((acc, cell, idx) => {
    if (cell === null) acc.push(idx);
    return acc;
  }, []);
}

/**
 * Standard Minimax algorithm without pruning.
 * Maximizing player = AI, minimizing player = Human.
 * @param {Array<string|null>} board - Current board state
 * @param {number} depth - Remaining search depth
 * @param {boolean} isMaximizing - True if it's the AI's turn
 * @param {string} aiSymbol - AI's symbol
 * @param {string} humanSymbol - Human's symbol
 * @param {{ count: number }} counter - Shared states-explored counter object
 * @returns {{ score: number }} Score of the best move
 */
function minimax(board, depth, isMaximizing, aiSymbol, humanSymbol, counter) {
  counter.count += 1;

  const score = evaluateBoard(board, aiSymbol, humanSymbol);
  if (score !== 0) return { score };
  if (getAvailableMoves(board).length === 0 || depth === 0) return { score: 0 };

  const moves = getAvailableMoves(board);

  if (isMaximizing) {
    let best = { score: -Infinity };
    for (const idx of moves) {
      board[idx] = aiSymbol;
      const result = minimax(board, depth - 1, false, aiSymbol, humanSymbol, counter);
      board[idx] = null;
      if (result.score > best.score) best = { score: result.score };
    }
    return best;
  } else {
    let best = { score: Infinity };
    for (const idx of moves) {
      board[idx] = humanSymbol;
      const result = minimax(board, depth - 1, true, aiSymbol, humanSymbol, counter);
      board[idx] = null;
      if (result.score < best.score) best = { score: result.score };
    }
    return best;
  }
}

/**
 * Minimax algorithm with Alpha-Beta Pruning.
 * Prunes branches where beta <= alpha (impossible to affect outcome).
 * @param {Array<string|null>} board - Current board state
 * @param {number} depth - Remaining search depth
 * @param {number} alpha - Best score maximizer can guarantee
 * @param {number} beta - Best score minimizer can guarantee
 * @param {boolean} isMaximizing - True if it's the AI's turn
 * @param {string} aiSymbol - AI's symbol
 * @param {string} humanSymbol - Human's symbol
 * @param {{ count: number }} counter - Shared states-explored counter object
 * @returns {{ score: number }} Score of the best move after pruning
 */
function minimaxAlphaBeta(board, depth, alpha, beta, isMaximizing, aiSymbol, humanSymbol, counter) {
  counter.count += 1;

  const score = evaluateBoard(board, aiSymbol, humanSymbol);
  if (score !== 0) return { score };
  if (getAvailableMoves(board).length === 0 || depth === 0) return { score: 0 };

  const moves = getAvailableMoves(board);

  if (isMaximizing) {
    let best = { score: -Infinity };
    for (const idx of moves) {
      board[idx] = aiSymbol;
      const result = minimaxAlphaBeta(board, depth - 1, alpha, beta, false, aiSymbol, humanSymbol, counter);
      board[idx] = null;
      if (result.score > best.score) best = { score: result.score };
      alpha = Math.max(alpha, best.score);
      if (beta <= alpha) break; // ← Pruning cut
    }
    return best;
  } else {
    let best = { score: Infinity };
    for (const idx of moves) {
      board[idx] = humanSymbol;
      const result = minimaxAlphaBeta(board, depth - 1, alpha, beta, true, aiSymbol, humanSymbol, counter);
      board[idx] = null;
      if (result.score < best.score) best = { score: result.score };
      beta = Math.min(beta, best.score);
      if (beta <= alpha) break; // ← Pruning cut
    }
    return best;
  }
}

/**
 * Selects the best move for the AI given the difficulty setting.
 * @param {Array<string|null>} board - Current board state
 * @param {string} difficultyKey - Key from DIFFICULTY_LEVELS
 * @param {string} aiSymbol - AI's symbol
 * @param {string} humanSymbol - Human's symbol
 * @returns {{ index: number, score: number, statesExplored: number, searchDepth: number, usedAlphaBeta: boolean, algorithm: string }}
 */
function getBestMove(board, difficultyKey, aiSymbol, humanSymbol) {
  const config = DIFFICULTY_LEVELS[difficultyKey];
  const moves = getAvailableMoves(board);
  if (moves.length === 0) return null;

  // EASY: Return random move — no computation
  if (difficultyKey === 'EASY') {
    const randomIdx = moves[Math.floor(Math.random() * moves.length)];
    return {
      index: randomIdx,
      score: 0,
      statesExplored: 1,
      searchDepth: 0,
      usedAlphaBeta: false,
      algorithm: 'Random Selection'
    };
  }

  const counter = { count: 0 };
  let bestScore = -Infinity;
  let bestIndex = moves[0];
  const boardCopy = [...board];

  for (const idx of moves) {
    boardCopy[idx] = aiSymbol;
    let result;
    if (config.useAlphaBeta) {
      result = minimaxAlphaBeta(boardCopy, config.depth - 1, -Infinity, Infinity, false, aiSymbol, humanSymbol, counter);
    } else {
      result = minimax(boardCopy, config.depth - 1, false, aiSymbol, humanSymbol, counter);
    }
    boardCopy[idx] = null;

    if (result.score > bestScore) {
      bestScore = result.score;
      bestIndex = idx;
    }
  }

  return {
    index: bestIndex,
    score: bestScore,
    statesExplored: counter.count,
    searchDepth: config.depth,
    usedAlphaBeta: config.useAlphaBeta,
    algorithm: config.useAlphaBeta
      ? 'Minimax + Alpha-Beta Pruning'
      : `Minimax (depth ${config.depth})`
  };
}

/**
 * Generates a human-readable explanation for why the AI chose a specific cell.
 * @param {Array<string|null>} board - Board state BEFORE the move was placed
 * @param {number} chosenIndex - The cell index chosen by the AI
 * @param {string} aiSymbol - AI's symbol
 * @param {string} humanSymbol - Human's symbol
 * @returns {string} One-sentence strategy explanation
 */
function getStrategyExplanation(board, chosenIndex, aiSymbol, humanSymbol) {
  const boardCopy = [...board];

  // Check if the move wins the game for AI
  boardCopy[chosenIndex] = aiSymbol;
  if (evaluateBoard(boardCopy, aiSymbol, humanSymbol) === 10) {
    return `AI selected Cell ${chosenIndex + 1} to claim victory — a decisive winning move!`;
  }
  boardCopy[chosenIndex] = null;

  // Check if the move blocks the human from winning
  for (const combo of WINNING_COMBINATIONS) {
    const [a, b, c] = combo;
    const cells = [board[a], board[b], board[c]];
    const humanCount = cells.filter(v => v === humanSymbol).length;
    const emptyCount = cells.filter(v => v === null).length;
    if (humanCount === 2 && emptyCount === 1) {
      const emptyIdx = [a, b, c][cells.indexOf(null)];
      if (emptyIdx === chosenIndex) {
        return `AI selected Cell ${chosenIndex + 1} to block your winning path — threat neutralized.`;
      }
    }
  }

  // Center play
  if (chosenIndex === 4) {
    return `AI selected the center (Cell 5) — statistically the strongest opening position.`;
  }

  // Corner play
  if ([0, 2, 6, 8].includes(chosenIndex)) {
    return `AI selected a corner cell (Cell ${chosenIndex + 1}) to maximize future strategic forking options.`;
  }

  return `AI selected Cell ${chosenIndex + 1} to maintain optimal board control and positional advantage.`;
}

// ===== GAME LOGIC =====

/**
 * Checks the board for a winner.
 * @param {Array<string|null>} board - Current board state
 * @returns {{ winner: string|null, combination: number[]|null }}
 */
function checkWinner(board) {
  for (const combo of WINNING_COMBINATIONS) {
    const [a, b, c] = combo;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return { winner: board[a], combination: combo };
    }
  }
  return { winner: null, combination: null };
}

/**
 * Checks if the game is a draw (all cells filled, no winner).
 * @param {Array<string|null>} board - Current board state
 * @returns {boolean}
 */
function checkDraw(board) {
  return board.every(cell => cell !== null) && checkWinner(board).winner === null;
}

// ===== HELPER UTILITIES =====

/** Returns cell label for a given index (e.g., 'center', 'corner', 'edge') */
function getCellLabel(idx) {
  if (idx === 4) return 'center';
  if ([0, 2, 6, 8].includes(idx)) return 'corner';
  return 'edge';
}

// ===== ROOT COMPONENT =====

export default function TicTacToeMaster() {
  // ── Game State ──────────────────────────────────────────────────────────────
  const [board, setBoard]                     = useState(Array(9).fill(null));
  const [currentPlayer, setCurrentPlayer]     = useState('X');
  const [winner, setWinner]                   = useState(null);         // 'X', 'O', 'draw'
  const [winningCells, setWinningCells]       = useState([]);
  const [gameMode, setGameMode]               = useState('HVA');
  const [difficulty, setDifficulty]           = useState('EXPERT');
  const [isAIThinking, setIsAIThinking]       = useState(false);
  const [aiStats, setAiStats]                 = useState(null);
  const [strategyInsight, setStrategyInsight] = useState('');
  const [moveHistory, setMoveHistory]         = useState([]);
  const [analytics, setAnalytics]             = useState({
    gamesPlayed: 0, aiWins: 0, humanWins: 0, draws: 0, gameDurations: []
  });
  const [gameStartTime, setGameStartTime]     = useState(() => Date.now());
  const [isEducationOpen, setIsEducationOpen] = useState(false);
  const [avaRunning, setAvaRunning]           = useState(false);
  const [displayedCount, setDisplayedCount]   = useState(0);
  const [mobileTab, setMobileTab]             = useState('game'); // 'game'|'stats'|'learn'
  const [leftPanelOpen, setLeftPanelOpen]     = useState(true);
  const [rightPanelOpen, setRightPanelOpen]   = useState(true);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const avaTimeoutRef    = useRef(null);
  const moveHistoryRef   = useRef(null);   // DOM ref for the scrollable list
  const historyDataRef   = useRef([]);      // data ref — always holds latest moveHistory array
  const boardRef         = useRef(board);
  const playerRef        = useRef(currentPlayer);
  const winnerRef        = useRef(winner);

  // Keep refs in sync
  useEffect(() => { boardRef.current = board; }, [board]);
  useEffect(() => { playerRef.current = currentPlayer; }, [currentPlayer]);
  useEffect(() => { winnerRef.current = winner; }, [winner]);
  useEffect(() => { historyDataRef.current = moveHistory; }, [moveHistory]);

  // ── Auto-scroll move history ──────────────────────────────────────────────
  useEffect(() => {
    if (moveHistoryRef.current) {
      moveHistoryRef.current.scrollTop = moveHistoryRef.current.scrollHeight;
    }
  }, [moveHistory]);

  // ── Animated states-explored counter ─────────────────────────────────────
  useEffect(() => {
    if (!aiStats) return;
    const target = aiStats.statesExplored;
    const duration = 600;
    const steps = 30;
    const increment = Math.ceil(target / steps);
    let current = 0;
    const interval = setInterval(() => {
      current = Math.min(current + increment, target);
      setDisplayedCount(current);
      if (current >= target) clearInterval(interval);
    }, duration / steps);
    return () => clearInterval(interval);
  }, [aiStats]);

  // ── Core: apply a move to the board ─────────────────────────────────────
  const applyMove = useCallback((boardState, index, player, isAI, currentMoveHistory) => {
    const newBoard = [...boardState];
    newBoard[index] = player;

    const { winner: w, combination } = checkWinner(newBoard);
    const isDraw = !w && checkDraw(newBoard);
    const gameOver = !!w || isDraw;

    const label = `${isAI ? '🤖 AI' : '👤 Human'} → Cell ${index + 1} [${getCellLabel(index)}]`;
    const newHistory = [
      ...currentMoveHistory,
      { moveNumber: currentMoveHistory.length + 1, player, cellIndex: index, label, isAI }
    ];

    return { newBoard, winner: w || (isDraw ? 'draw' : null), winningCells: combination || [], gameOver, newHistory };
  }, []);

  // ── Finalize game outcome ─────────────────────────────────────────────────
  const finalizeGame = useCallback((outcome, startTime) => {
    setAnalytics(prev => {
      const duration = (Date.now() - startTime) / 1000;
      return {
        gamesPlayed: prev.gamesPlayed + 1,
        aiWins:      outcome === 'O' ? prev.aiWins + 1 : prev.aiWins,    // AI is always O in HVA
        humanWins:   outcome === 'X' ? prev.humanWins + 1 : prev.humanWins,
        draws:       outcome === 'draw' ? prev.draws + 1 : prev.draws,
        gameDurations: [...prev.gameDurations, duration]
      };
    });
  }, []);

  // ── Human cell click ──────────────────────────────────────────────────────
  const handleCellClick = useCallback((index) => {
    if (winner || board[index] || isAIThinking) return;

    // In HVA mode, only allow clicks on human turn (X = human)
    if (gameMode === 'HVA' && currentPlayer !== 'X') return;
    // In AVA mode, no human clicks
    if (gameMode === 'AVA') return;

    // Apply human move
    const { newBoard, winner: w, winningCells: wc, gameOver, newHistory } = applyMove(
      board, index, currentPlayer, false, moveHistory
    );

    setBoard(newBoard);
    setMoveHistory(newHistory);

    if (gameOver) {
      setWinner(w);
      setWinningCells(wc);
      finalizeGame(w, gameStartTime);
      return;
    }

    const nextPlayer = currentPlayer === 'X' ? 'O' : 'X';

    // HVH — just switch player
    if (gameMode === 'HVH') {
      setCurrentPlayer(nextPlayer);
      return;
    }

    // HVA — trigger AI move after delay
    setCurrentPlayer(nextPlayer);
    setIsAIThinking(true);
    setAiStats(null);
    setStrategyInsight('');

    // Capture board/history for use inside setTimeout (no stale refs)
    const boardSnap   = newBoard;
    const historySnap = newHistory;

    setTimeout(() => {
      const move = getBestMove(boardSnap, difficulty, 'O', 'X');
      if (!move) { setIsAIThinking(false); return; }

      const explanation = getStrategyExplanation(boardSnap, move.index, 'O', 'X');
      setAiStats(move);
      setStrategyInsight(explanation);

      const { newBoard: afterAI, winner: aiWinner, winningCells: aiWC, gameOver: aiGameOver, newHistory: histAfterAI } = applyMove(
        boardSnap, move.index, 'O', true, historySnap
      );

      setBoard(afterAI);
      setMoveHistory(histAfterAI);
      setIsAIThinking(false);

      if (aiGameOver) {
        setWinner(aiWinner);
        setWinningCells(aiWC);
        finalizeGame(aiWinner, gameStartTime);
      } else {
        setCurrentPlayer('X');
      }
    }, DELAY_MS.AI_THINK);
  }, [board, winner, isAIThinking, currentPlayer, gameMode, difficulty, moveHistory, gameStartTime, applyMove, finalizeGame]);

  // ── New Game ──────────────────────────────────────────────────────────────
  const handleNewGame = useCallback(() => {
    if (avaTimeoutRef.current) clearTimeout(avaTimeoutRef.current);
    setBoard(Array(9).fill(null));
    setCurrentPlayer('X');
    setWinner(null);
    setWinningCells([]);
    setIsAIThinking(false);
    setAiStats(null);
    setStrategyInsight('');
    setMoveHistory([]);
    setAvaRunning(false);
    setGameStartTime(Date.now());
  }, []);

  // ── Reset Stats ───────────────────────────────────────────────────────────
  const handleResetStats = useCallback(() => {
    setAnalytics({ gamesPlayed: 0, aiWins: 0, humanWins: 0, draws: 0, gameDurations: [] });
  }, []);

  // ── Game mode change ──────────────────────────────────────────────────────
  const handleModeChange = useCallback((mode) => {
    if (avaTimeoutRef.current) clearTimeout(avaTimeoutRef.current);
    setGameMode(mode);
    setBoard(Array(9).fill(null));
    setCurrentPlayer('X');
    setWinner(null);
    setWinningCells([]);
    setIsAIThinking(false);
    setAiStats(null);
    setStrategyInsight('');
    setMoveHistory([]);
    setAvaRunning(false);
    setGameStartTime(Date.now());
  }, []);

  // ── AVA Loop ─────────────────────────────────────────────────────────────
  // Uses refs to read current state synchronously, then dispatches a single
  // atomic setState batch to avoid stale-closure / nested-setState issues.
  const runAVAStep = useCallback(() => {
    const currentWinner = winnerRef.current;
    if (currentWinner) { setAvaRunning(false); return; }

    const snap_board    = boardRef.current;
    const snap_player   = playerRef.current;
    const aiSym         = snap_player;
    const humanSym      = snap_player === 'X' ? 'O' : 'X';

    setIsAIThinking(true);
    setAiStats(null);
    setStrategyInsight('');

    avaTimeoutRef.current = setTimeout(() => {
      // Read latest refs (in case a stop was triggered during delay)
      if (winnerRef.current) { setIsAIThinking(false); setAvaRunning(false); return; }

      const move = getBestMove(snap_board, difficulty, aiSym, humanSym);
      if (!move) { setIsAIThinking(false); setAvaRunning(false); return; }

      const explanation = getStrategyExplanation(snap_board, move.index, aiSym, humanSym);
      setAiStats(move);
      setStrategyInsight(explanation);

      const currentHistory = historyDataRef.current ?? [];
      const { newBoard, winner: w, winningCells: wc, gameOver, newHistory } = applyMove(
        snap_board, move.index, aiSym, true, currentHistory
      );

      // Atomic state updates
      setBoard(newBoard);
      setMoveHistory(newHistory);
      setCurrentPlayer(humanSym);
      setIsAIThinking(false);

      if (gameOver) {
        setWinner(w);
        setWinningCells(wc);
        finalizeGame(w, gameStartTime);
        setAvaRunning(false);
        // Auto-restart after 2 s
        setTimeout(() => {
          handleNewGame();
          setTimeout(() => setAvaRunning(true), 400);
        }, 2000);
      } else {
        // Schedule the next move
        avaTimeoutRef.current = setTimeout(() => runAVAStepRef.current(), DELAY_MS.AVA_MOVE);
      }
    }, DELAY_MS.AI_THINK);
  }, [difficulty, applyMove, finalizeGame, gameStartTime, handleNewGame]);

  // Keep runAVAStep ref fresh
  const runAVAStepRef = useRef(runAVAStep);
  useEffect(() => { runAVAStepRef.current = runAVAStep; }, [runAVAStep]);

  const handleStartAVA = useCallback(() => {
    if (avaRunning) {
      if (avaTimeoutRef.current) clearTimeout(avaTimeoutRef.current);
      setAvaRunning(false);
      setIsAIThinking(false);
      return;
    }
    handleNewGame();
    setAvaRunning(true);
    setTimeout(() => runAVAStepRef.current(), 300);
  }, [avaRunning, handleNewGame]);

  // AVA effect trigger when avaRunning turns true
  useEffect(() => {
    if (avaRunning && gameMode === 'AVA' && !winner) {
      // Running is managed by runAVAStep chaining
    }
  }, [avaRunning, gameMode, winner]);

  // ── Derived values ────────────────────────────────────────────────────────
  const aiSymbol    = 'O';
  const humanSymbol = 'X';
  const isGameOver  = !!winner;
  const pruningEff  = aiStats && aiStats.usedAlphaBeta
    ? ((1 - aiStats.statesExplored / MAX_STATES) * 100).toFixed(1)
    : null;
  const avgDuration = analytics.gameDurations.length > 0
    ? (analytics.gameDurations.reduce((a, b) => a + b, 0) / analytics.gameDurations.length).toFixed(1)
    : '—';
  const aiWinRate = analytics.gamesPlayed > 0
    ? ((analytics.aiWins / analytics.gamesPlayed) * 100).toFixed(1)
    : '0.0';

  // ── Algorithm flow steps ──────────────────────────────────────────────────
  const FLOW_STEPS = [
    { icon: '📋', label: 'Current Board State' },
    { icon: '🌿', label: 'Generate Possible Moves' },
    { icon: '🔍', label: 'Evaluate Future States' },
    { icon: '⚡', label: 'Apply Minimax Algorithm' },
    { icon: '✂️', label: 'Alpha-Beta Pruning' },
    { icon: '🏆', label: 'Select Best Move' },
    { icon: '▶️', label: 'Execute Move' },
  ];

  // ============================================================
  // Component: AppHeader — sticky top navigation bar
  // ============================================================
  const AppHeader = () => (
    <header className="sticky top-0 z-50 border-b border-white/8 bg-black/60 backdrop-blur-xl">
      <div className="max-w-screen-xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="relative flex-shrink-0">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center shadow-lg shadow-cyan-500/30 float-anim">
              <Zap size={20} className="text-white" />
            </div>
          </div>
          <div className="min-w-0">
            <h1 className="text-lg font-black tracking-tight gradient-text-cyan whitespace-nowrap">
              Tic-Tac-Toe AI Master
            </h1>
            <p className="text-xs text-gray-600 font-mono tracking-wide hidden sm:block">
              Minimax · Alpha-Beta Pruning · Game Theory
            </p>
          </div>
        </div>

        {/* Center Badge */}
        <div className="hidden md:flex items-center gap-2">
          <span className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold tracking-wide">
            {GAME_MODES[gameMode]}
          </span>
          {isAIThinking && (
            <span className="px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/30 text-violet-400 text-xs font-semibold pulse-glow-anim">
              AI Thinking...
            </span>
          )}
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={() => setIsEducationOpen(p => !p)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
              isEducationOpen
                ? 'bg-violet-500/20 border border-violet-500/40 text-violet-400'
                : 'bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:border-white/20'
            }`}
          >
            <BookOpen size={15} />
            <span className="hidden sm:inline">Education</span>
          </button>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-center w-9 h-9 rounded-lg bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:border-white/20 transition-all duration-200"
          >
            <Code2 size={16} />
          </a>
        </div>
      </div>
    </header>
  );

  // ============================================================
  // Component: GameModeSelector — toggle between game modes
  // ============================================================
  const GameModeSelector = () => {
    const modes = [
      { key: 'HVA', label: 'Human vs AI',    icon: <><User size={13}/><span className="mx-0.5 text-gray-500">vs</span><Bot size={13}/></> },
      { key: 'HVH', label: 'Human vs Human', icon: <><User size={13}/><span className="mx-0.5 text-gray-500">vs</span><User size={13}/></> },
      { key: 'AVA', label: 'AI vs AI',       icon: <><Bot size={13}/><span className="mx-0.5 text-gray-500">vs</span><Bot size={13}/></> },
    ];
    return (
      <div className="space-y-2">
        <p className="text-xs tracking-widest text-gray-500 uppercase font-semibold">Game Mode</p>
        <div className="space-y-1.5">
          {modes.map(({ key, label, icon }) => (
            <button
              key={key}
              onClick={() => handleModeChange(key)}
              className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-sm font-semibold transition-all duration-200 ${
                gameMode === key
                  ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-400 shadow-sm shadow-cyan-500/20'
                  : 'bg-white/3 border-white/8 text-gray-400 hover:text-white hover:border-white/20 hover:bg-white/5'
              }`}
            >
              <span className="flex items-center gap-1">{icon}</span>
              <span>{label}</span>
              {gameMode === key && <span className="ml-auto w-2 h-2 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400/50" />}
            </button>
          ))}
        </div>
      </div>
    );
  };

  // ============================================================
  // Component: DifficultySelector — difficulty pill buttons
  // ============================================================
  const DifficultySelector = () => {
    if (gameMode === 'HVH') return null;
    const levels = [
      { key: 'EASY',   color: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10 shadow-emerald-500/20' },
      { key: 'MEDIUM', color: 'text-amber-400 border-amber-500/40 bg-amber-500/10 shadow-amber-500/20' },
      { key: 'HARD',   color: 'text-orange-400 border-orange-500/40 bg-orange-500/10 shadow-orange-500/20' },
      { key: 'EXPERT', color: 'text-red-400 border-red-500/40 bg-red-500/10 shadow-red-500/20' },
    ];
    return (
      <div className="space-y-2">
        <p className="text-xs tracking-widest text-gray-500 uppercase font-semibold">AI Difficulty</p>
        <div className="grid grid-cols-2 gap-1.5">
          {levels.map(({ key, color }) => (
            <button
              key={key}
              onClick={() => setDifficulty(key)}
              className={`px-2 py-2 rounded-lg border text-xs font-bold transition-all duration-200 ${
                difficulty === key
                  ? `${color} shadow-sm`
                  : 'text-gray-500 border-white/8 bg-white/3 hover:text-gray-300 hover:border-white/15'
              }`}
            >
              {DIFFICULTY_LEVELS[key].label}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500 italic leading-relaxed">
          {DIFFICULTY_LEVELS[difficulty].description}
        </p>
      </div>
    );
  };

  // ============================================================
  // Component: MoveHistoryPanel — scrollable move list
  // ============================================================
  const MoveHistoryPanel = () => (
    <div className="space-y-2 flex-1 min-h-0">
      <p className="text-xs tracking-widest text-gray-500 uppercase font-semibold flex items-center gap-2">
        <Hash size={11} /> Move History
      </p>
      <div
        ref={moveHistoryRef}
        className="overflow-y-auto space-y-1 max-h-48 pr-1"
      >
        {moveHistory.length === 0 ? (
          <p className="text-xs text-gray-600 italic text-center py-4">
            No moves yet. Start playing!
          </p>
        ) : (
          moveHistory.map((move) => (
            <div
              key={move.moveNumber}
              className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium slide-in-up-anim ${
                move.isAI
                  ? 'bg-cyan-500/8 border border-cyan-500/15 text-cyan-300'
                  : 'bg-violet-500/8 border border-violet-500/15 text-violet-300'
              }`}
            >
              <span className="text-gray-600 font-mono w-5 text-right flex-shrink-0">#{move.moveNumber}</span>
              <span className="flex-1 truncate">{move.label}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );

  // ============================================================
  // Component: LearningOutcomePanel — concept badges
  // ============================================================
  const LearningOutcomePanel = () => {
    const [open, setOpen] = useState(true);
    const topics = [
      { label: 'Artificial Intelligence', color: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/8' },
      { label: 'Game Theory',             color: 'text-violet-400 border-violet-500/30 bg-violet-500/8' },
      { label: 'Search Algorithms',       color: 'text-blue-400 border-blue-500/30 bg-blue-500/8' },
      { label: 'Minimax Algorithm',       color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/8' },
      { label: 'Alpha-Beta Pruning',      color: 'text-amber-400 border-amber-500/30 bg-amber-500/8' },
      { label: 'Decision Trees',          color: 'text-orange-400 border-orange-500/30 bg-orange-500/8' },
      { label: 'Strategic Thinking',      color: 'text-red-400 border-red-500/30 bg-red-500/8' },
    ];
    return (
      <div className="space-y-2">
        <button
          onClick={() => setOpen(p => !p)}
          className="flex items-center justify-between w-full text-xs tracking-widest text-gray-500 uppercase font-semibold"
        >
          <span className="flex items-center gap-2"><Sparkles size={11} /> What You'll Learn</span>
          {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
        {open && (
          <div className="flex flex-wrap gap-1.5">
            {topics.map(({ label, color }) => (
              <span
                key={label}
                className={`px-2 py-0.5 rounded-full border text-xs font-medium ${color}`}
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  // ============================================================
  // Component: LeftPanel — game config + history
  // ============================================================
  const LeftPanel = () => (
    <aside className="glass-panel p-4 flex flex-col gap-5 h-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain size={16} className="text-cyan-400" />
          <span className="text-sm font-bold text-white">Game Setup</span>
        </div>
      </div>
      <GameModeSelector />
      <DifficultySelector />
      <div className="border-t border-white/8" />
      <MoveHistoryPanel />
      <div className="border-t border-white/8" />
      <LearningOutcomePanel />
    </aside>
  );

  // ============================================================
  // Component: StatusBar — game status message
  // ============================================================
  const StatusBar = () => {
    let content;
    if (winner === 'draw') {
      content = (
        <div className="flex items-center gap-2 text-amber-400 fade-in-anim">
          <Handshake size={20} />
          <span className="text-lg font-black tracking-wide">It's a Draw!</span>
        </div>
      );
    } else if (winner) {
      const isAIWin = winner === aiSymbol && gameMode !== 'HVH';
      content = (
        <div className={`flex items-center gap-2 fade-in-anim ${isAIWin ? 'text-cyan-400' : 'text-violet-400'}`}>
          <Trophy size={22} className="float-anim" />
          <span className="text-lg font-black tracking-wide">
            {gameMode === 'HVH'
              ? `Player ${winner} Wins!`
              : isAIWin ? '🤖 AI Wins!' : '🏆 You Win!'}
          </span>
        </div>
      );
    } else if (isAIThinking) {
      content = (
        <div className="flex items-center gap-2 text-cyan-400 pulse-glow-anim px-4 py-1 rounded-xl">
          <Cpu size={18} className="animate-spin" />
          <span className="text-base font-bold">
            AI is Thinking<span className="thinking-dots" />
          </span>
        </div>
      );
    } else {
      const isHumanTurn = gameMode === 'HVH' || (gameMode === 'HVA' && currentPlayer === humanSymbol);
      content = (
        <div className={`flex items-center gap-2 ${
          isHumanTurn ? 'text-violet-400' : 'text-cyan-400'
        }`}>
          {isHumanTurn ? <User size={18} /> : <Bot size={18} />}
          <span className="text-base font-semibold">
            {gameMode === 'AVA'
              ? `AI ${currentPlayer} is ready`
              : isHumanTurn
                ? `Your Turn — Place ${currentPlayer}`
                : `AI is ready...`}
          </span>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-center min-h-12">
        {content}
      </div>
    );
  };

  // ============================================================
  // Component: GameBoard — 3×3 interactive grid
  // ============================================================
  const GameBoard = () => (
    <div className="relative">
      {/* Decorative glow behind board */}
      <div className="absolute inset-0 -m-4 rounded-3xl bg-gradient-to-br from-cyan-500/5 via-transparent to-violet-500/5 blur-xl pointer-events-none" />
      <div className="relative grid grid-cols-3 gap-2 p-2">
        {board.map((cell, idx) => {
          const isWinCell  = winningCells.includes(idx);
          const isEmpty    = cell === null;
          const canClick   = isEmpty && !isGameOver && !isAIThinking &&
            (gameMode === 'HVH' || (gameMode === 'HVA' && currentPlayer === humanSymbol));

          return (
            <button
              key={idx}
              onClick={() => canClick && handleCellClick(idx)}
              className={`
                relative aspect-square flex items-center justify-center rounded-2xl border
                transition-all duration-150 group select-none
                ${isWinCell
                  ? 'winning-cell-anim'
                  : isEmpty && canClick
                    ? 'bg-white/4 border-white/10 hover:bg-white/10 hover:border-cyan-500/40 hover:scale-105'
                    : 'bg-white/3 border-white/8'
                }
                ${!canClick && isEmpty ? 'cursor-not-allowed opacity-40' : ''}
                ${!isEmpty ? 'cursor-default' : canClick ? 'cursor-pointer' : ''}
              `}
              style={{ minHeight: '90px' }}
            >
              {/* Cell index hint */}
              {isEmpty && (
                <span className="absolute top-1.5 right-2 text-xs text-gray-700 font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                  {idx + 1}
                </span>
              )}

              {/* Cell symbol */}
              {cell && (
                <span
                  className={`cell-pop-anim text-4xl md:text-5xl font-black leading-none select-none ${
                    cell === 'X' ? 'text-violet-400' : 'text-cyan-400'
                  }`}
                  style={{
                    textShadow: cell === 'X'
                      ? '0 0 20px rgba(167, 139, 250, 0.6)'
                      : '0 0 20px rgba(34, 211, 238, 0.6)'
                  }}
                >
                  {cell}
                </span>
              )}

              {/* Hover glow */}
              {isEmpty && canClick && (
                <span className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-br from-cyan-500/5 to-violet-500/5" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );

  // ============================================================
  // Component: AIThinkingDisplay — live computation card
  // ============================================================
  const AIThinkingDisplay = () => {
    if (!isAIThinking && !aiStats) return null;
    const stats = aiStats;

    return (
      <div className={`glass-panel p-4 border-l-2 border-cyan-500/60 slide-in-up-anim ${isAIThinking ? 'pulse-glow-anim' : ''}`}>
        <div className="flex items-center gap-2 mb-3">
          <Cpu size={16} className={`text-cyan-400 ${isAIThinking ? 'animate-spin' : ''}`} />
          <span className="text-sm font-bold text-cyan-400">
            {isAIThinking ? 'AI is Computing...' : 'Computation Complete'}
          </span>
        </div>
        <div className="space-y-2 font-mono text-xs">
          <div className="flex justify-between items-center">
            <span className="text-gray-500">Algorithm</span>
            <span className="text-cyan-300 font-semibold">{stats?.algorithm || DIFFICULTY_LEVELS[difficulty].description}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-500">Search Depth</span>
            <span className="text-violet-300 font-semibold">{stats?.searchDepth ?? DIFFICULTY_LEVELS[difficulty].depth} levels</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-500">States Explored</span>
            <span className="text-emerald-300 font-bold counter-up-anim">
              {stats ? displayedCount.toLocaleString() : '—'}
            </span>
          </div>
          {stats?.usedAlphaBeta && pruningEff && (
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Pruning Efficiency</span>
              <span className="text-amber-300 font-bold">{pruningEff}%</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  // ============================================================
  // Component: AIStrategyInsight — why the AI chose this cell
  // ============================================================
  const AIStrategyInsight = () => {
    if (!strategyInsight) return null;
    return (
      <div className="glass-panel p-4 border-l-2 border-violet-500/60 fade-in-anim">
        <div className="flex items-center gap-2 mb-2">
          <Brain size={15} className="text-violet-400" />
          <span className="text-xs font-bold text-violet-400 uppercase tracking-widest">AI Strategy Insight</span>
        </div>
        <p className="text-sm text-gray-200 leading-relaxed">{strategyInsight}</p>
        {aiStats && (
          <div className="mt-2 flex items-center gap-2">
            <span className="px-2 py-0.5 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300 text-xs font-mono">
              Best Move Score: {aiStats.score > 0 ? '+' : ''}{aiStats.score}
            </span>
          </div>
        )}
      </div>
    );
  };

  // ============================================================
  // Component: GameControls — action buttons
  // ============================================================
  const GameControls = () => (
    <div className="flex gap-2 flex-wrap">
      <button
        onClick={handleNewGame}
        className={`flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 text-sm font-bold transition-all duration-200 hover:bg-emerald-500/25 hover:scale-105 ${isGameOver ? 'new-game-pulse' : ''}`}
      >
        <RefreshCw size={15} />
        New Game
      </button>
      <button
        onClick={handleResetStats}
        className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-amber-500/15 border border-amber-500/40 text-amber-400 text-sm font-bold transition-all duration-200 hover:bg-amber-500/25 hover:scale-105"
      >
        <RotateCcw size={15} />
        Reset Stats
      </button>
      {gameMode === 'AVA' && (
        <button
          onClick={handleStartAVA}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 hover:scale-105 ${
            avaRunning
              ? 'bg-red-500/15 border border-red-500/40 text-red-400 hover:bg-red-500/25'
              : 'bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/25'
          }`}
        >
          {avaRunning ? <><Square size={15} /> Stop AI</> : <><Play size={15} /> Start AI vs AI</>}
        </button>
      )}
    </div>
  );

  // ============================================================
  // Component: AnalyticsDashboard — right panel stats grid
  // ============================================================
  const AnalyticsDashboard = () => {
    const stats = [
      { label: 'Games Played', value: analytics.gamesPlayed, color: 'text-cyan-400' },
      { label: 'AI Wins',       value: analytics.aiWins,      color: 'text-red-400' },
      { label: 'Human Wins',    value: analytics.humanWins,   color: 'text-violet-400' },
      { label: 'Draws',         value: analytics.draws,       color: 'text-amber-400' },
      { label: 'AI Win Rate',   value: `${aiWinRate}%`,       color: 'text-emerald-400' },
      { label: 'Avg Duration',  value: `${avgDuration}s`,     color: 'text-blue-400' },
    ];

    const total = analytics.aiWins + analytics.humanWins + analytics.draws || 1;
    const bars = [
      { label: 'AI',    value: analytics.aiWins,    pct: analytics.aiWins / total * 100,    color: 'bg-cyan-500' },
      { label: 'Human', value: analytics.humanWins, pct: analytics.humanWins / total * 100, color: 'bg-violet-500' },
      { label: 'Draw',  value: analytics.draws,     pct: analytics.draws / total * 100,     color: 'bg-amber-500' },
    ];

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <BarChart3 size={15} className="text-cyan-400" />
          <span className="text-xs font-bold text-white uppercase tracking-widest">Game Analytics</span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {stats.map(({ label, value, color }) => (
            <div key={label} className="bg-white/3 border border-white/8 rounded-xl p-2.5 text-center">
              <div className={`font-mono text-xl font-bold ${color}`}>{value}</div>
              <div className="text-xs text-gray-500 uppercase tracking-wide mt-0.5">{label}</div>
            </div>
          ))}
        </div>

        {/* Win/Loss/Draw bar chart */}
        <div className="space-y-1.5">
          {bars.map(({ label, value, pct, color }) => (
            <div key={label} className="flex items-center gap-2 text-xs">
              <span className="text-gray-500 w-10 text-right">{label}</span>
              <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${color} transition-all duration-700`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-gray-500 w-4 font-mono">{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ============================================================
  // Component: AIThinkingVisualizer — computation breakdown
  // ============================================================
  const AIThinkingVisualizer = () => {
    if (!aiStats) return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Cpu size={15} className="text-cyan-400" />
          <span className="text-xs font-bold text-white uppercase tracking-widest">AI Computation</span>
        </div>
        <p className="text-xs text-gray-600 italic">Make a move to see AI computation stats.</p>
      </div>
    );

    const relBar = Math.min((aiStats.statesExplored / MAX_STATES) * 100, 100);

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Activity size={15} className="text-cyan-400" />
          <span className="text-xs font-bold text-white uppercase tracking-widest">AI Computation</span>
        </div>

        {/* Algorithm badge */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`px-2 py-0.5 rounded-full border text-xs font-semibold ${
            aiStats.usedAlphaBeta
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
              : 'bg-violet-500/15 border-violet-500/40 text-violet-300'
          }`}>
            {aiStats.algorithm}
          </span>
        </div>

        {/* States explored */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-gray-500">States Explored</span>
            <span className="text-emerald-400 font-mono font-bold">{aiStats.statesExplored.toLocaleString()}</span>
          </div>
          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all duration-700"
              style={{ width: `${Math.max(relBar, 1)}%` }}
            />
          </div>
          <div className="text-xs text-gray-600 font-mono">of {MAX_STATES.toLocaleString()} max possible</div>
        </div>

        {/* Pruning efficiency */}
        {pruningEff && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Pruning Efficiency</span>
              <span className="text-amber-400 font-mono font-bold">{pruningEff}%</span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-orange-400 rounded-full transition-all duration-700"
                style={{ width: `${pruningEff}%` }}
              />
            </div>
          </div>
        )}

        {/* Decision */}
        <div className="bg-white/3 border border-white/8 rounded-lg p-2.5">
          <div className="text-xs text-gray-500 mb-0.5">Decision</div>
          <div className="text-xs text-cyan-300 font-mono font-semibold">
            Best move → Cell {aiStats.index + 1} (Score: {aiStats.score > 0 ? '+' : ''}{aiStats.score})
          </div>
        </div>

        {/* Tree depth visualization (4 rows of dots) */}
        <div className="space-y-1">
          <div className="text-xs text-gray-500">Search Tree Depth</div>
          {[1, 3, 7, 9].map((nodes, rowIdx) => (
            <div key={rowIdx} className="flex items-center gap-1">
              <span className="text-xs text-gray-700 font-mono w-4">L{rowIdx + 1}</span>
              <div className="flex gap-1 flex-wrap">
                {Array(Math.min(nodes, 9)).fill(0).map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 rounded-full transition-colors ${
                      i < Math.ceil(nodes * (aiStats.statesExplored / MAX_STATES + 0.3))
                        ? 'bg-cyan-500/70'
                        : 'bg-white/10'
                    }`}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ============================================================
  // Component: AlgorithmFlowDiagram — step-by-step flow
  // ============================================================
  const AlgorithmFlowDiagram = () => (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <GitBranch size={15} className="text-violet-400" />
        <span className="text-xs font-bold text-white uppercase tracking-widest">Algorithm Flow</span>
      </div>
      <div className="space-y-1">
        {FLOW_STEPS.map((step, idx) => (
          <div key={idx} className="flex flex-col items-center">
            <div
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all duration-300 ${
                isAIThinking
                  ? `flow-step-active flow-delay-${idx} border-cyan-500/40 bg-cyan-500/10 text-cyan-300`
                  : aiStats && idx <= 5
                    ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-300'
                    : 'border-white/8 bg-white/3 text-gray-500'
              }`}
            >
              <span>{step.icon}</span>
              <span>{step.label}</span>
              {aiStats && !isAIThinking && idx <= 5 && (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
              )}
            </div>
            {idx < FLOW_STEPS.length - 1 && (
              <div className={`w-px h-2 transition-colors ${
                isAIThinking ? 'bg-cyan-500/50' : aiStats ? 'bg-emerald-500/30' : 'bg-white/8'
              }`} />
            )}
          </div>
        ))}
      </div>
    </div>
  );

  // ============================================================
  // Component: RightPanel — analytics + AI visualization
  // ============================================================
  const RightPanel = () => (
    <aside className="glass-panel p-4 flex flex-col gap-5 h-full overflow-y-auto">
      <AnalyticsDashboard />
      <div className="border-t border-white/8" />
      <AIThinkingVisualizer />
      <div className="border-t border-white/8" />
      <AlgorithmFlowDiagram />
    </aside>
  );

  // ============================================================
  // Component: GameCenter — board + status + controls
  // ============================================================
  const GameCenter = () => (
    <div className="flex flex-col items-center gap-4">
      {/* Status */}
      <div className="w-full">
        <StatusBar />
      </div>

      {/* Board */}
      <div className="w-full max-w-sm">
        <GameBoard />
      </div>

      {/* AI info cards */}
      <div className="w-full max-w-sm space-y-3">
        <AIThinkingDisplay />
        <AIStrategyInsight />
      </div>

      {/* Controls */}
      <div className="w-full max-w-sm">
        <GameControls />
      </div>
    </div>
  );

  // ============================================================
  // Component: EducationPanel — 6 AI concept cards
  // ============================================================
  const EducationPanel = () => {
    if (!isEducationOpen) return null;

    const cards = [
      {
        icon: '🤖',
        title: 'What is Artificial Intelligence?',
        color: 'border-cyan-500/30 bg-cyan-500/5',
        accent: 'text-cyan-400',
        content: `Artificial Intelligence (AI) is the field of computer science focused on building systems that can perform tasks requiring human-like intelligence — including decision-making, pattern recognition, language understanding, and strategic planning.

In this app, the AI uses search algorithms to "think ahead" and evaluate thousands of possible future game states before making a single move. Key subfields demonstrated here include Search & Optimization, Game Theory, and Decision-Making under constraints.`
      },
      {
        icon: '♟️',
        title: 'What is Game Theory?',
        color: 'border-violet-500/30 bg-violet-500/5',
        accent: 'text-violet-400',
        content: `Game Theory is the mathematical study of strategic interactions between rational decision-makers. In zero-sum games like Tic-Tac-Toe, one player's gain is exactly the other player's loss.

Key concepts: Nash Equilibrium (no player can improve by unilaterally changing strategy), optimal strategies (each player plays to maximize their minimum guaranteed outcome), and minimax principle — the foundation of game-playing AI. Expert mode demonstrates a provably optimal Nash Equilibrium strategy.`
      },
      {
        icon: '🌲',
        title: 'What is the Minimax Algorithm?',
        color: 'border-emerald-500/30 bg-emerald-500/5',
        accent: 'text-emerald-400',
        content: `Minimax is a recursive search algorithm used in two-player zero-sum games. The AI (Maximizer) tries to maximize its score; the opponent (Minimizer) tries to minimize the AI's score. Both are assumed to play optimally.

Pseudo-code:
  if maximizing:
    pick the move with the HIGHEST score
  if minimizing:
    pick the move with the LOWEST score

Scores: AI win = +10, Human win = -10, Draw = 0. The algorithm explores all future board states up to a terminal condition (win/draw/depth limit) and selects the move that leads to the best guaranteed outcome.`
      },
      {
        icon: '✂️',
        title: 'What is Alpha-Beta Pruning?',
        color: 'border-amber-500/30 bg-amber-500/5',
        accent: 'text-amber-400',
        content: `Alpha-Beta Pruning is an optimization of Minimax that eliminates branches of the search tree that cannot possibly influence the final decision. It maintains two values:
• Alpha (α): best score the maximizer can guarantee
• Beta (β): best score the minimizer can guarantee
When β ≤ α, the branch is "pruned" (skipped entirely).

Complexity: Minimax is O(b^d); Alpha-Beta reduces this to O(b^(d/2)) in the best case — equivalent to doubling the search depth for free! Compare Hard vs Expert mode in this app: Expert explores dramatically fewer states while reaching identical decisions.`
      },
      {
        icon: '🔍',
        title: 'What is State Space Search?',
        color: 'border-blue-500/30 bg-blue-500/5',
        accent: 'text-blue-400',
        content: `The State Space is the set of all possible configurations (states) a system can be in. In Tic-Tac-Toe, a state is one specific arrangement of X's, O's, and empty cells on the 3×3 board.

Total possible states: up to 9! = 362,880 (though many are unreachable). Minimax performs a Depth-First Search (DFS) through this state space, systematically exploring all valid game paths from the current board position. The "States Explored" counter you see in this app counts exactly how many nodes in this state space tree the algorithm visited before selecting its move.`
      },
      {
        icon: '🧠',
        title: 'How Does the AI Make Decisions?',
        color: 'border-red-500/30 bg-red-500/5',
        accent: 'text-red-400',
        content: `The AI decision pipeline:
1. 📋 Read current board state
2. 🌿 Generate all legal moves (empty cells)
3. 🔍 For each move, recursively evaluate all resulting future states
4. ⚡ Apply Minimax: propagate scores back up the tree
5. ✂️ Alpha-Beta Pruning: skip provably suboptimal branches
6. 🏆 Select the move with the highest guaranteed score
7. ▶️ Execute and display result

The evaluateBoard() function scores terminal states: +10 for AI win, -10 for human win, 0 for draw. This numerical grounding is what lets the algorithm compare abstract "which future is better?" across thousands of possible game continuations.`
      }
    ];

    return (
      <div className="border-t border-white/8 slide-in-up-anim">
        <div className="max-w-screen-xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen size={20} className="text-violet-400" />
            <h2 className="text-xl font-black text-white">AI Education Center</h2>
            <span className="px-2 py-0.5 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-400 text-xs font-semibold">6 Concepts</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cards.map((card, idx) => (
              <div
                key={idx}
                className={`glass-panel border p-5 ${card.color} slide-in-up-anim`}
                style={{ animationDelay: `${idx * 80}ms` }}
              >
                <div className="flex items-center gap-3 mb-3">
                  <span className="text-2xl">{card.icon}</span>
                  <h3 className={`text-sm font-bold ${card.accent} leading-tight`}>{card.title}</h3>
                </div>
                <p className="text-xs text-gray-400 leading-relaxed whitespace-pre-line">{card.content}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // ============================================================
  // Mobile bottom tab bar
  // ============================================================
  // Component: MobileEducationCards — education cards for mobile learn tab
  // ============================================================
  const MobileEducationCards = () => {
    const cards = [
      { icon: '🤖', title: 'Artificial Intelligence', color: 'border-cyan-500/30', accent: 'text-cyan-400',
        content: 'AI builds systems that perform tasks requiring human-like intelligence — decision-making, search, and strategic planning. This app uses search algorithms to evaluate thousands of future game states in real time.' },
      { icon: '♟️', title: 'Game Theory', color: 'border-violet-500/30', accent: 'text-violet-400',
        content: 'The mathematical study of strategic interactions. In zero-sum games like Tic-Tac-Toe, one player\'s gain equals the other\'s loss. Expert mode proves a Nash Equilibrium — the optimal strategy from which no player can deviate and improve.' },
      { icon: '🌲', title: 'Minimax Algorithm', color: 'border-emerald-500/30', accent: 'text-emerald-400',
        content: 'Minimax recursively explores the game tree. The AI (Maximizer) picks the move with the highest score; the opponent (Minimizer) picks the lowest. Scores: AI win = +10, Human win = -10, Draw = 0.' },
      { icon: '✂️', title: 'Alpha-Beta Pruning', color: 'border-amber-500/30', accent: 'text-amber-400',
        content: 'Prunes game tree branches that can\'t affect the outcome. Reduces complexity from O(b^d) to O(b^(d/2)). Compare Hard vs Expert mode — Expert explores far fewer states but makes identical decisions.' },
      { icon: '🔍', title: 'State Space Search', color: 'border-blue-500/30', accent: 'text-blue-400',
        content: 'The state space is all possible board configurations — up to 9! = 362,880 in Tic-Tac-Toe. Minimax performs depth-first search through this space. The "States Explored" counter shows exactly how many nodes the algorithm visits.' },
      { icon: '🧠', title: 'AI Decision Pipeline', color: 'border-red-500/30', accent: 'text-red-400',
        content: 'Board → Generate legal moves → Recursively evaluate states → Apply Minimax → Alpha-Beta pruning → Select highest-score move → Execute. The evaluateBoard() function provides the numerical grounding: +10, -10, or 0.' },
    ];
    return (
      <div className="px-3 py-4 space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <BookOpen size={16} className="text-violet-400" />
          <h2 className="text-base font-black text-white">AI Education Center</h2>
        </div>
        {cards.map((card, idx) => (
          <div key={idx} className={`glass-panel border p-4 ${card.color} slide-in-up-anim`} style={{ animationDelay: `${idx * 60}ms` }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">{card.icon}</span>
              <h3 className={`text-sm font-bold ${card.accent}`}>{card.title}</h3>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">{card.content}</p>
          </div>
        ))}
      </div>
    );
  };

  // ============================================================
  const MobileTabBar = () => (
    <div className="md:hidden fixed bottom-0 left-0 right-0 z-40 border-t border-white/8 bg-black/80 backdrop-blur-xl">
      <div className="flex">
        {[
          { key: 'game',  icon: <Target size={18} />, label: 'Game' },
          { key: 'stats', icon: <TrendingUp size={18} />, label: 'Stats' },
          { key: 'learn', icon: <BookOpen size={18} />, label: 'Learn' },
        ].map(({ key, icon, label }) => (
          <button
            key={key}
            onClick={() => setMobileTab(key)}
            className={`flex-1 flex flex-col items-center gap-1 py-3 text-xs font-semibold transition-colors ${
              mobileTab === key ? 'text-cyan-400' : 'text-gray-600'
            }`}
          >
            {icon}
            {label}
          </button>
        ))}
      </div>
    </div>
  );

  // ============================================================
  // MAIN RENDER
  // ============================================================
  return (
    <div className="min-h-screen grid-bg text-white">
      <AppHeader />

      {/* Decorative floating orbs */}
      <div className="fixed top-1/4 left-10 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none float-anim" />
      <div className="fixed bottom-1/4 right-10 w-64 h-64 bg-violet-500/5 rounded-full blur-3xl pointer-events-none float-anim" style={{ animationDelay: '1.5s' }} />

      {/* ── Desktop Layout: 3 columns ── */}
      <main className="hidden md:block max-w-screen-xl mx-auto px-4 py-6">
        <div className="grid grid-cols-[280px_1fr_280px] gap-5 items-start">
          {/* Left Panel */}
          <div className="sticky top-24">
            <LeftPanel />
          </div>

          {/* Center */}
          <GameCenter />

          {/* Right Panel */}
          <div className="sticky top-24">
            <RightPanel />
          </div>
        </div>

        {/* Education Panel */}
        <EducationPanel />
      </main>

      {/* ── Mobile Layout: tab-based ── */}
      <div className="md:hidden pb-20">
        {mobileTab === 'game' && (
          <div className="px-3 py-4 space-y-4">
            {/* Condensed mode/difficulty row */}
            <div className="glass-panel p-3">
              <div className="flex gap-2 flex-wrap mb-2">
                {Object.keys(GAME_MODES).map(key => (
                  <button
                    key={key}
                    onClick={() => handleModeChange(key)}
                    className={`px-2.5 py-1 rounded-lg border text-xs font-bold transition-all ${
                      gameMode === key
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                        : 'bg-white/3 border-white/8 text-gray-500'
                    }`}
                  >
                    {key}
                  </button>
                ))}
              </div>
              {gameMode !== 'HVH' && (
                <div className="flex gap-1.5 flex-wrap">
                  {Object.keys(DIFFICULTY_LEVELS).map(key => (
                    <button
                      key={key}
                      onClick={() => setDifficulty(key)}
                      className={`px-2.5 py-1 rounded-lg border text-xs font-bold transition-all ${
                        difficulty === key ? 'bg-red-500/20 border-red-500/50 text-red-400' : 'bg-white/3 border-white/8 text-gray-500'
                      }`}
                    >
                      {DIFFICULTY_LEVELS[key].label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <GameCenter />

            {/* Move history */}
            <div className="glass-panel p-3">
              <MoveHistoryPanel />
            </div>
          </div>
        )}

        {mobileTab === 'stats' && (
          <div className="px-3 py-4 space-y-4">
            <div className="glass-panel p-4">
              <AnalyticsDashboard />
            </div>
            <div className="glass-panel p-4">
              <AIThinkingVisualizer />
            </div>
            <div className="glass-panel p-4">
              <AlgorithmFlowDiagram />
            </div>
          </div>
        )}

        {mobileTab === 'learn' && (
          <MobileEducationCards />
        )}
      </div>

      {/* Mobile tab bar */}
      <MobileTabBar />
    </div>
  );
}
