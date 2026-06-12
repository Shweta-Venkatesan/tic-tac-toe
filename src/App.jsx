// =============================================================================
// TicTacToeMaster.jsx — Complete AI-powered Tic-Tac-Toe (single file)
// AI Concepts: Minimax, Alpha-Beta Pruning, Game Theory, State Space Search
// Author: AI Internship Portfolio Project
// =============================================================================

import { useState, useEffect, useCallback, useRef } from 'react';

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
  // ── Global Styles & Fonts (Injected) ────────────────────────────────────────
  useEffect(() => {
    // Bootstrap Icons
    if (!document.querySelector('link[href*="bootstrap-icons"]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css';
      document.head.appendChild(link);
    }
    // JetBrains Mono
    if (!document.querySelector('link[href*="JetBrains+Mono"]')) {
      const font = document.createElement('link');
      font.href = 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap';
      font.rel = 'stylesheet';
      document.head.appendChild(font);
    }
  }, []);

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
  // CSS & Keyframes
  // ============================================================
  const GlobalStyles = () => (
    <style dangerouslySetInnerHTML={{ __html: `
      :root {
        --bg-base:         #07080f;
        --bg-surface:      #0c0d18;
        --bg-elevated:     #111221;
        --bg-overlay:      #171829;
        --bg-panel:        #0e0f1e;

        --border-subtle:   rgba(255,255,255,0.05);
        --border-default:  rgba(255,255,255,0.09);
        --border-strong:   rgba(255,255,255,0.16);
        --border-accent:   rgba(6,182,212,0.35);

        --cyan:            #06b6d4;
        --cyan-dim:        rgba(6,182,212,0.12);
        --cyan-glow:       rgba(6,182,212,0.30);
        --cyan-bright:     #22d3ee;

        --violet:          #8b5cf6;
        --violet-dim:      rgba(139,92,246,0.12);
        --violet-glow:     rgba(139,92,246,0.30);

        --emerald:         #10b981;
        --emerald-dim:     rgba(16,185,129,0.12);
        --amber:           #f59e0b;
        --amber-dim:       rgba(245,158,11,0.12);
        --red:             #ef4444;
        --red-dim:         rgba(239,68,68,0.12);

        --text-primary:    #e8e9f5;
        --text-secondary:  #8b8ca8;
        --text-muted:      #4a4b66;
        --text-cyan:       #67e8f9;
        --text-violet:     #c4b5fd;

        --x-color:         #8b5cf6;
        --x-glow:          rgba(139,92,246,0.40);
        --o-color:         #06b6d4;
        --o-glow:          rgba(6,182,212,0.40);

        --radius-sm:       4px;
        --radius-md:       8px;
        --radius-lg:       12px;
        --radius-xl:       16px;
      }

      /* Typography Rules */
      .app-title { font-size: 1rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
      .section-header { font-size: 0.6rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: var(--text-muted); }
      .panel-heading { font-size: 0.85rem; font-weight: 700; letter-spacing: -0.2px; color: var(--text-primary); }
      .data-value { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 1.6rem; font-weight: 700; }
      .ui-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--text-muted); }
      .body-text { font-size: 0.78rem; line-height: 1.65; color: var(--text-secondary); }
      .mono-inline { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-cyan); background: var(--bg-overlay); padding: 2px 7px; border-radius: 4px; border: 1px solid var(--border-subtle); }

      /* Keyframe Animations */
      @keyframes cellPlace {
        0%   { opacity: 0; transform: scale(0.3) rotate(-15deg); }
        60%  { transform: scale(1.12) rotate(3deg); opacity: 1; }
        100% { transform: scale(1) rotate(0deg); opacity: 1; }
      }
      @keyframes winFlash {
        0%, 100% { background: var(--emerald-dim); border-color: rgba(16,185,129,0.40); }
        50%      { background: rgba(16,185,129,0.25); border-color: var(--emerald); }
      }
      @keyframes scanline {
        0%   { transform: translateY(-100%); opacity: 0.06; }
        100% { transform: translateY(100%);  opacity: 0.06; }
      }
      @keyframes thinkBar {
        0%, 100% { transform: scaleY(0.3); opacity: 0.3; }
        50%      { transform: scaleY(1);   opacity: 1; }
      }
      @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 0 0 var(--cyan-glow); }
        50%      { box-shadow: 0 0 20px 4px var(--cyan-glow); }
      }
      @keyframes flowStep {
        0%   { opacity: 0; transform: translateX(-8px); }
        100% { opacity: 1; transform: translateX(0); }
      }
      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes counterTick {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes borderDraw {
        from { stroke-dashoffset: 200; }
        to   { stroke-dashoffset: 0; }
      }
      
      .cell-anim-place { animation: cellPlace 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards; }
      .cell-anim-win { animation: winFlash 0.8s ease-in-out infinite; border-color: var(--emerald) !important; box-shadow: inset 0 0 12px rgba(16,185,129,0.20); }
      .anim-fade-up { animation: fadeUp 0.3s ease forwards; }
      .anim-flow-step { animation: flowStep 0.3s ease forwards; }
      .anim-glow-pulse { animation: glowPulse 1.5s infinite; }
    `}} />
  );

  // ============================================================
  // Component: AppHeader
  // ============================================================
  const AppHeader = () => (
    <header style={{
      height: '56px',
      background: 'var(--bg-surface)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      position: 'sticky',
      top: 0,
      zIndex: 50
    }}>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <i className="bi bi-cpu-fill" style={{ color: 'var(--cyan)', fontSize: '1.05rem' }} />
        <span className="app-title" style={{ color: 'var(--text-primary)', marginLeft: '12px' }}>TIC-TAC-TOE</span>
        <span style={{
          color: 'var(--cyan)', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '3px',
          marginLeft: '8px', background: 'var(--cyan-dim)', border: '1px solid var(--border-accent)',
          borderRadius: '4px', padding: '2px 8px', textTransform: 'uppercase'
        }}>AI MASTER</span>
      </div>

      <div className="hidden md:flex" style={{ display: 'flex', gap: '8px' }}>
        {[
          { icon: 'bi-diagram-2-fill', label: 'MINIMAX', color: 'var(--cyan)' },
          { icon: 'bi-scissors', label: 'ALPHA-BETA', color: 'var(--violet)' },
          { icon: 'bi-diagram-3-fill', label: 'GAME THEORY', color: 'var(--text-muted)' }
        ].map((badge, i) => (
          <div key={i} style={{
            background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)', padding: '4px 10px', fontSize: '0.62rem',
            fontWeight: 700, letterSpacing: '1.5px', display: 'flex', alignItems: 'center', gap: '6px'
          }}>
            <i className={`bi ${badge.icon}`} style={{ color: badge.color }} />
            <span style={{ color: badge.color }}>{badge.label}</span>
          </div>
        ))}
      </div>

      <div>
        <button
          onClick={() => setIsEducationOpen(p => !p)}
          style={{
            width: '34px', height: '34px', background: isEducationOpen ? 'var(--violet-dim)' : 'var(--bg-elevated)',
            border: `1px solid ${isEducationOpen ? 'var(--violet-glow)' : 'var(--border-default)'}`,
            borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: isEducationOpen ? 'var(--violet)' : 'var(--text-secondary)', cursor: 'pointer',
            transition: 'all 0.15s'
          }}
          onMouseEnter={(e) => {
            if (!isEducationOpen) {
              e.currentTarget.style.background = 'var(--violet-dim)';
              e.currentTarget.style.borderColor = 'var(--violet-glow)';
              e.currentTarget.style.color = 'var(--violet)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isEducationOpen) {
              e.currentTarget.style.background = 'var(--bg-elevated)';
              e.currentTarget.style.borderColor = 'var(--border-default)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }
          }}
        >
          <i className="bi bi-journal-code" />
        </button>
      </div>
    </header>
  );

  // ============================================================
  // Component: LeftPanel
  // ============================================================
  const LeftPanel = () => {
    const modes = [
      { key: 'HVA', iconL: 'bi-person-fill', iconLCol: 'var(--violet)', iconR: 'bi-cpu-fill', iconRCol: 'var(--cyan)', label: 'HUMAN vs AI', sub: 'Play against the AI' },
      { key: 'HVH', iconL: 'bi-person-fill', iconLCol: 'var(--violet)', iconR: 'bi-person-fill', iconRCol: 'var(--violet)', label: 'HUMAN vs HUMAN', sub: 'Local multiplayer' },
      { key: 'AVA', iconL: 'bi-cpu-fill', iconLCol: 'var(--cyan)', iconR: 'bi-cpu-fill', iconRCol: 'var(--cyan)', label: 'AI vs AI', sub: 'Watch AI battle' }
    ];

    const diffs = [
      { key: 'EASY', icon: 'bi-reception-1', color: '#10b981', borderAct: 'var(--emerald)', sub: 'Random moves' },
      { key: 'MEDIUM', icon: 'bi-reception-3', color: '#f59e0b', borderAct: 'var(--amber)', sub: '3-level lookahead' },
      { key: 'HARD', icon: 'bi-reception-4', color: '#f97316', borderAct: '#f97316', sub: 'Full Minimax' },
      { key: 'EXPERT', icon: 'bi-lightning-charge-fill', color: '#ef4444', borderAct: 'var(--red)', sub: 'Alpha-Beta Pruning' }
    ];

    const concepts = [
      { label: 'Artificial Intelligence', icon: 'bi-robot', color: 'var(--cyan)' },
      { label: 'Game Theory', icon: 'bi-diagram-2-fill', color: 'var(--violet)' },
      { label: 'Search Algorithms', icon: 'bi-search', color: '#3b82f6' },
      { label: 'Minimax', icon: 'bi-diagram-3-fill', color: 'var(--emerald)' },
      { label: 'Alpha-Beta Pruning', icon: 'bi-scissors', color: 'var(--amber)' },
      { label: 'Decision Trees', icon: 'bi-bezier2', color: '#f97316' },
      { label: 'Strategic Thinking', icon: 'bi-lightbulb-fill', color: 'var(--red)' }
    ];

    return (
      <aside className="h-full overflow-y-auto pb-5">
        {/* Game Mode */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '18px 16px 10px', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '3px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
          <i className="bi bi-controller" /> GAME MODE
        </div>
        {modes.map(m => {
          const isActive = gameMode === m.key;
          return (
            <div key={m.key} onClick={() => handleModeChange(m.key)}
              style={{
                display: 'flex', alignItems: 'center', gap: '12px', padding: '11px 14px', margin: '0 10px 5px',
                background: isActive ? 'var(--cyan-dim)' : 'var(--bg-elevated)',
                border: '1px solid',
                borderColor: isActive ? 'var(--border-accent)' : 'var(--border-subtle)',
                borderLeft: isActive ? '2px solid var(--cyan)' : '1px solid var(--border-accent)',
                borderRadius: 'var(--radius-md)', cursor: 'pointer', transition: 'all 0.15s'
              }}
            >
              <div style={{ width: '32px', height: '32px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-overlay)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.9rem', gap: '2px' }}>
                <i className={`bi ${m.iconL}`} style={{ color: m.iconLCol, fontSize: '0.7rem' }} />
                <i className="bi bi-slash-lg" style={{ color: 'var(--text-muted)', fontSize: '0.5rem' }} />
                <i className={`bi ${m.iconR}`} style={{ color: m.iconRCol, fontSize: '0.7rem' }} />
              </div>
              <div>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.5px' }}>{m.label}</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{m.sub}</div>
              </div>
            </div>
          );
        })}

        <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '10px 16px' }} />

        {/* Difficulty */}
        {(gameMode === 'HVA' || gameMode === 'AVA') && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 16px', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '3px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              <i className="bi bi-speedometer2" /> DIFFICULTY
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', padding: '0 16px' }}>
              {diffs.map(d => {
                const isActive = difficulty === d.key;
                return (
                  <div key={d.key} onClick={() => setDifficulty(d.key)}
                    style={{
                      padding: '10px 12px', background: isActive ? `${d.color}20` : 'var(--bg-elevated)',
                      border: `1px solid ${isActive ? d.borderAct : 'var(--border-subtle)'}`,
                      borderRadius: 'var(--radius-md)', display: 'flex', flexDirection: 'column', gap: '4px',
                      cursor: 'pointer', transition: 'all 0.15s'
                    }}
                  >
                    <i className={`bi ${d.icon}`} style={{ fontSize: '0.9rem', color: d.color }} />
                    <div style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '1.5px', color: 'var(--text-primary)' }}>{d.key}</div>
                    <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>{d.sub}</div>
                  </div>
                );
              })}
            </div>
            <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '10px 16px' }} />
          </>
        )}

      </aside>
    );
  };

  // ============================================================
  // Component: GameCenter
  // ============================================================
  const GameCenter = () => {
    
    const getStatusProps = () => {
      if (winner && winner !== 'draw') {
        const pName = gameMode === 'HVH' ? `PLAYER ${winner}` : (winner === 'O' ? 'AI' : 'HUMAN');
        return {
          leftIcon: 'bi-trophy-fill', leftColor: 'var(--amber)', leftText: `${pName} WINS`,
          rightContent: <i className="bi bi-star-fill" style={{ color: 'var(--amber)' }} />,
          bg: 'var(--amber-dim)', borderL: '2px solid var(--amber)'
        };
      }
      if (winner === 'draw') {
        return {
          leftIcon: 'bi-dash-circle-fill', leftColor: 'var(--text-secondary)', leftText: 'DRAW — PERFECT PLAY',
          rightContent: <i className="bi bi-slash-lg" style={{ color: 'var(--text-muted)' }} />,
          bg: 'var(--bg-elevated)', borderL: '2px solid var(--text-muted)'
        };
      }
      if (isAIThinking) {
        return {
          leftIcon: 'bi-cpu-fill', leftColor: 'var(--cyan)', leftText: 'AI COMPUTING', leftAnim: '',
          rightContent: (
            <div style={{ display: 'flex', gap: '2px', alignItems: 'flex-end', height: '14px' }}>
              {[1,2,3,4,5].map(i => <div key={i} style={{ width: '3px', background: 'var(--cyan)', animation: `thinkBar 0.8s ease-in-out infinite`, animationDelay: `${i*0.1}s` }} />)}
            </div>
          ),
          bg: 'var(--cyan-dim)', borderL: '2px solid var(--cyan)'
        };
      }
      
      const isHuman = currentPlayer === 'X' || gameMode === 'HVH';
      return {
        leftIcon: isHuman ? 'bi-person-fill' : 'bi-cpu-fill',
        leftColor: isHuman ? 'var(--violet)' : 'var(--cyan)',
        leftText: gameMode === 'AVA' ? 'AI READY' : 'YOUR TURN',
        rightContent: currentPlayer === 'X' ? <i className="bi bi-x-lg" style={{ color: 'var(--x-color)' }} /> : <i className="bi bi-circle" style={{ color: 'var(--o-color)' }} />,
        bg: isHuman ? 'var(--violet-dim)' : 'var(--cyan-dim)',
        borderL: isHuman ? '2px solid var(--violet)' : '2px solid var(--cyan)'
      };
    };

    const st = getStatusProps();

    return (
      <div className="flex-1 flex flex-col items-center py-8 px-4 md:px-8 bg-[var(--bg-base)] h-full overflow-y-auto">
        {/* Status Bar */}
        <div style={{ width: '100%', maxWidth: '380px', background: st.bg, border: '1px solid var(--border-default)', borderLeft: st.borderL, borderRadius: 'var(--radius-lg)', padding: '12px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <i className={`bi ${st.leftIcon} ${st.leftAnim || ''}`} style={{ color: st.leftColor }} />
            <span style={{ color: st.leftColor, fontSize: '0.75rem', fontWeight: 700, letterSpacing: '1px' }}>{st.leftText}</span>
          </div>
          <div>{st.rightContent}</div>
        </div>

        {/* Game Board */}
        <div style={{ width: '100%', maxWidth: '320px', aspectRatio: '1/1', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gridTemplateRows: 'repeat(3, 1fr)', gap: '6px', background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)', padding: '10px', position: 'relative' }}>
          <div style={{ position: 'absolute', inset: 0, background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.015) 2px, rgba(255,255,255,0.015) 4px)', pointerEvents: 'none', borderRadius: 'inherit' }} />
          
          {board.map((cell, idx) => {
            const isWin = winningCells.includes(idx);
            const isDisabled = !!winner || isAIThinking || (gameMode === 'HVA' && currentPlayer === 'O') || (gameMode === 'AVA');
            
            return (
              <div key={idx} onClick={() => handleCellClick(idx)}
                style={{
                  background: cell === 'X' ? 'var(--violet-dim)' : cell === 'O' ? 'var(--cyan-dim)' : 'var(--bg-elevated)',
                  border: `1px solid ${cell === 'X' ? 'rgba(139,92,246,0.25)' : cell === 'O' ? 'rgba(6,182,212,0.25)' : 'var(--border-subtle)'}`,
                  borderRadius: 'var(--radius-lg)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: cell || isDisabled ? 'not-allowed' : 'pointer', transition: 'all 0.15s', position: 'relative', overflow: 'hidden',
                  opacity: isDisabled && !cell ? 0.5 : 1
                }}
                className={isWin ? 'cell-anim-win' : ''}
                onMouseEnter={(e) => { if(!cell && !isDisabled) { e.currentTarget.style.background = 'var(--bg-overlay)'; e.currentTarget.style.borderColor = 'var(--border-default)'; } }}
                onMouseLeave={(e) => { if(!cell && !isDisabled) { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.borderColor = 'var(--border-subtle)'; } }}
              >
                {!cell && <div style={{ position: 'absolute', bottom: '5px', right: '7px', fontSize: '0.55rem', fontWeight: 600, color: 'var(--text-muted)', fontFamily: 'monospace' }}>{idx}</div>}
                {cell === 'X' && <i className="bi bi-x-lg cell-anim-place" style={{ fontSize: '2.4rem', fontWeight: 900, color: 'var(--x-color)', filter: 'drop-shadow(0 0 8px var(--x-glow))' }} />}
                {cell === 'O' && <i className="bi bi-circle cell-anim-place" style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--o-color)', filter: 'drop-shadow(0 0 8px var(--o-glow))' }} />}
              </div>
            );
          })}
        </div>

        {/* AI Insight */}
        <div style={{ width: '100%', maxWidth: '380px', minHeight: '110px', marginTop: '16px' }}>
          {isAIThinking ? (
            <div className="anim-fade-up" style={{ height: '100%', background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderLeft: '2px solid var(--cyan)', borderRadius: 'var(--radius-lg)', padding: '14px 18px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
              <div style={{ display: 'flex', gap: '4px', alignItems: 'flex-end', height: '20px' }}>
                {[1,2,3,4,5].map(i => <div key={i} style={{ width: '4px', background: 'var(--cyan)', animation: `thinkBar 0.8s ease-in-out infinite`, animationDelay: `${i*0.1}s`, borderRadius: '2px' }} />)}
              </div>
              <span style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '2px', color: 'var(--cyan)' }}>COMPUTING OPTIMAL MOVE...</span>
            </div>
          ) : strategyInsight ? (
            <div className="anim-fade-up" style={{ height: '100%', background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderLeft: '2px solid var(--cyan)', borderRadius: 'var(--radius-lg)', padding: '14px 18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <i className="bi bi-cpu-fill" style={{ color: 'var(--cyan)', fontSize: '0.85rem' }} />
                <span style={{ fontSize: '0.62rem', fontWeight: 700, letterSpacing: '2.5px', color: 'var(--cyan)' }}>AI INSIGHT</span>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: '8px' }}>{strategyInsight}</div>
              {aiStats && (
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginTop: '8px', background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '3px 10px' }}>
                  <i className="bi bi-graph-up-arrow" style={{ color: 'var(--emerald)', fontSize: '0.7rem' }} />
                  <span style={{ fontFamily: 'monospace', color: 'var(--text-cyan)', fontSize: '0.7rem' }}>SCORE: {aiStats.score > 0 ? '+' : ''}{aiStats.score}</span>
                </div>
              )}
            </div>
          ) : null}
        </div>

        {/* Game Controls */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '20px', width: '100%', maxWidth: '380px' }}>
          <button onClick={handleNewGame} style={{ flex: 1, background: 'var(--emerald-dim)', border: '1px solid rgba(16,185,129,0.30)', color: 'var(--emerald)', borderRadius: 'var(--radius-md)', padding: '10px', fontSize: '0.72rem', fontWeight: 700, letterSpacing: '1px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', cursor: 'pointer', transition: 'all 0.15s' }}>
            <i className="bi bi-arrow-counterclockwise" /> NEW GAME
          </button>
          <button onClick={handleResetStats} style={{ flex: 1, background: 'var(--amber-dim)', border: '1px solid rgba(245,158,11,0.30)', color: 'var(--amber)', borderRadius: 'var(--radius-md)', padding: '10px', fontSize: '0.72rem', fontWeight: 700, letterSpacing: '1px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', cursor: 'pointer', transition: 'all 0.15s' }}>
            <i className="bi bi-bootstrap-reboot" /> RESET STATS
          </button>
          {gameMode === 'AVA' && (
            <button onClick={handleStartAVA} style={{ flex: 1, background: 'var(--cyan-dim)', border: '1px solid var(--border-accent)', color: 'var(--cyan)', borderRadius: 'var(--radius-md)', padding: '10px', fontSize: '0.72rem', fontWeight: 700, letterSpacing: '1px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', cursor: 'pointer', transition: 'all 0.15s' }}>
              <i className={`bi ${avaRunning ? 'bi-stop-fill' : 'bi-play-fill'}`} /> {avaRunning ? 'STOP AI' : 'START AI'}
            </button>
          )}
        </div>
      </div>
    );
  };

  // ============================================================
  // Component: RightPanel
  // ============================================================
  const RightPanel = () => {
    const tiles = [
      { label: 'GAMES', icon: 'bi-controller', color: 'var(--cyan)', val: analytics.gamesPlayed },
      { label: 'AI WINS', icon: 'bi-cpu-fill', color: 'var(--cyan)', val: analytics.aiWins },
      { label: 'HUMAN WINS', icon: 'bi-person-fill', color: 'var(--violet)', val: analytics.humanWins },
      { label: 'DRAWS', icon: 'bi-dash-circle-fill', color: 'var(--text-muted)', val: analytics.draws },
      { label: 'WIN RATE', icon: 'bi-pie-chart-fill', color: 'var(--emerald)', val: `${aiWinRate}%` },
      { label: 'AVG TIME', icon: 'bi-stopwatch-fill', color: 'var(--amber)', val: `${avgDuration}s` }
    ];

    const tot = analytics.gamesPlayed || 1;
    const barW = (analytics.humanWins / tot) * 100;
    const barD = (analytics.draws / tot) * 100;
    const barA = (analytics.aiWins / tot) * 100;

    return (
      <aside className="h-full overflow-y-auto pb-5">
        
        {/* Analytics */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '18px 16px 10px', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '3px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
          <i className="bi bi-bar-chart-line-fill" /> ANALYTICS
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', padding: '0 12px 12px' }}>
          {tiles.map((t, i) => (
            <div key={i} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '12px 14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                <i className={`bi ${t.icon}`} style={{ fontSize: '0.85rem', color: t.color }} />
                <span style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>{t.label}</span>
              </div>
              <div style={{ fontFamily: 'monospace', fontSize: '1.5rem', fontWeight: 700, color: t.color }}>{t.val}</div>
            </div>
          ))}
        </div>
        <div style={{ height: '6px', background: 'var(--bg-overlay)', borderRadius: '3px', overflow: 'hidden', margin: '0 12px 12px', display: 'flex' }}>
          <div style={{ width: `${barW}%`, background: 'var(--violet)' }} />
          <div style={{ width: `${barD}%`, background: 'var(--text-muted)' }} />
          <div style={{ width: `${barA}%`, background: 'var(--cyan)' }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 14px', fontSize: '0.6rem', fontFamily: 'monospace', fontWeight: 700 }}>
          <span style={{ color: 'var(--violet)' }}>HUMAN {Math.round(barW)}%</span>
          <span style={{ color: 'var(--text-muted)' }}>DRAW {Math.round(barD)}%</span>
          <span style={{ color: 'var(--cyan)' }}>AI {Math.round(barA)}%</span>
        </div>

        <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '16px 16px' }} />

        {/* AI Computation Panel */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0 16px 10px', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '3px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
          <i className="bi bi-cpu-fill" /> AI COMPUTATION
        </div>
        <div style={{ padding: '0 12px 12px', minHeight: '160px' }}>
          {isAIThinking ? (
            <div className="anim-fade-up" style={{ height: '100%', background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-lg)', padding: '24px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
              <div style={{ display: 'flex', gap: '4px', alignItems: 'flex-end', height: '24px' }}>
                {[1,2,3,4,5].map(i => <div key={i} style={{ width: '4px', background: 'var(--cyan)', animation: `thinkBar 0.8s ease-in-out infinite`, animationDelay: `${i*0.1}s`, borderRadius: '2px' }} />)}
              </div>
              <span style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '2px', color: 'var(--cyan)' }}>ANALYZING GAME TREE</span>
            </div>
          ) : aiStats ? (
            <div className="anim-fade-up" style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-lg)', padding: '14px 16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <i className="bi bi-diagram-3-fill" style={{ color: 'var(--cyan)' }} />
                  <span style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>ALGORITHM</span>
                </div>
                <span style={{ background: 'var(--cyan-dim)', border: '1px solid var(--border-accent)', color: 'var(--text-cyan)', fontSize: '0.68rem', fontWeight: 700, borderRadius: 'var(--radius-sm)', padding: '3px 9px' }}>{aiStats.algorithm}</span>
              </div>
              <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '8px 0' }} />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <i className="bi bi-diagram-2-fill" style={{ color: 'var(--violet)' }} />
                  <span style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>STATES</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                  <span style={{ fontFamily: 'monospace', color: 'var(--text-violet)', fontWeight: 700 }}>{displayedCount.toLocaleString()}</span>
                  <div style={{ width: '40px', height: '3px', background: 'var(--bg-overlay)', borderRadius: '2px', overflow: 'hidden', marginTop: '2px' }}>
                    <div style={{ height: '100%', width: `${(aiStats.statesExplored / 362880) * 100}%`, background: 'var(--violet)' }} />
                  </div>
                </div>
              </div>
              <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '8px 0' }} />
              
              {aiStats.usedAlphaBeta && (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <i className="bi bi-scissors" style={{ color: 'var(--amber)' }} />
                      <span style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>PRUNED</span>
                    </div>
                    <span style={{ fontFamily: 'monospace', color: 'var(--amber)', fontWeight: 700 }}>{pruningEff}%</span>
                  </div>
                  <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '8px 0' }} />
                </>
              )}

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <i className="bi bi-bullseye" style={{ color: 'var(--emerald)' }} />
                  <span style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '2px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>DECISION</span>
                </div>
                <span style={{ fontFamily: 'monospace', color: 'var(--text-cyan)' }}>CELL {aiStats.index} · SCORE {aiStats.score > 0 ? '+' : ''}{aiStats.score}</span>
              </div>
            </div>
          ) : (
             <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-lg)', padding: '24px 16px', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
               <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '1px', fontWeight: 600 }}>WAITING FOR GAME</span>
             </div>
          )}
        </div>

      </aside>

    );
  };

  // ============================================================
  // Component: EducationPanel
  // ============================================================
  const EducationPanel = () => {
    if (!isEducationOpen) return null;

    const cards = [
      {
        icon: 'bi-robot', color: 'var(--cyan)', title: 'Artificial Intelligence',
        content: 'AI is the field of computer science focused on building systems that can perform tasks requiring human-like intelligence.\n\nIn this app, the AI uses search algorithms to "think ahead" and evaluate thousands of possible future game states before making a single move.'
      },
      {
        icon: 'bi-diagram-2-fill', color: 'var(--violet)', title: 'Game Theory',
        content: 'The mathematical study of strategic interactions between rational decision-makers. In zero-sum games like Tic-Tac-Toe, one player\'s gain is exactly the other player\'s loss.\n\nKey concepts: Nash Equilibrium, optimal strategies, and the minimax principle.'
      },
      {
        icon: 'bi-diagram-3-fill', color: 'var(--emerald)', title: 'Minimax Algorithm',
        content: 'Minimax is a recursive search algorithm. The Maximizer tries to maximize its score; the Minimizer tries to minimize it.\n',
        code: `if maximizing:
  return max(child scores)
if minimizing:
  return min(child scores)`
      },
      {
        icon: 'bi-scissors', color: 'var(--amber)', title: 'Alpha-Beta Pruning',
        content: 'An optimization of Minimax that eliminates branches that cannot possibly influence the final decision.\n',
        complexity: true
      },
      {
        icon: 'bi-search', color: '#3b82f6', title: 'State Space Search',
        content: 'The State Space is the set of all possible configurations a system can be in. Minimax performs a Depth-First Search (DFS) through this state space.\n\nThe "States Explored" counter shows exactly how many nodes the algorithm visited.'
      },
      {
        icon: 'bi-cpu-fill', color: 'var(--cyan)', title: 'How AI Decides',
        content: 'The algorithm evaluates terminal states: +10 for AI win, -10 for human win, 0 for draw. This numerical grounding lets it compare abstract futures across thousands of possible game continuations.'
      }
    ];

    return (
      <div className="anim-fade-up" style={{ background: 'var(--bg-surface)', borderTop: '1px solid var(--border-subtle)', padding: '24px 28px', gridColumn: '1 / -1' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <i className="bi bi-journal-code" style={{ color: 'var(--cyan)', fontSize: '1.2rem' }} />
            <span style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '2px', color: 'var(--text-primary)' }}>TECHNICAL REFERENCE</span>
          </div>
          <button onClick={() => setIsEducationOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <i className="bi bi-chevron-up" style={{ fontSize: '1.2rem' }} />
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
          {cards.map((c, i) => (
            <div key={i} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-lg)', padding: '20px 22px', transition: 'border-color 0.2s' }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--border-default)'}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-subtle)'}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', background: `${c.color}20`, borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <i className={`bi ${c.icon}`} style={{ color: c.color, fontSize: '1.1rem' }} />
                </div>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{c.title}</div>
              </div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.65, whiteSpace: 'pre-wrap' }}>
                {c.content}
                {c.code && (
                  <pre style={{ background: 'var(--bg-overlay)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '10px 12px', fontFamily: 'monospace', fontSize: '0.7rem', color: 'var(--text-cyan)', marginTop: '10px', overflowX: 'auto' }}>
                    {c.code}
                  </pre>
                )}
                {c.complexity && (
                  <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ fontFamily: 'monospace', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)' }}>
                      <i className="bi bi-x-circle" style={{ color: 'var(--red)' }} /> Without pruning : O(b^d)
                    </div>
                    <div style={{ fontFamily: 'monospace', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)' }}>
                      <i className="bi bi-check-circle-fill" style={{ color: 'var(--emerald)' }} /> With pruning    : O(b^(d/2))
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ============================================================
  // MAIN RENDER
  // ============================================================
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)', color: 'var(--text-primary)', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <GlobalStyles />
      <AppHeader />
      
      {/* Fully Responsive Container */}
      <div className="flex flex-col md:flex-row min-h-[calc(100vh-56px)] w-full relative">
        <div className="w-full md:w-1/4 border-b md:border-b-0 md:border-r border-[var(--border-subtle)] bg-[var(--bg-panel)] h-auto md:h-[calc(100vh-56px)] md:sticky md:top-[56px] overflow-hidden">
          <LeftPanel />
        </div>
        <div className="w-full md:w-2/4 bg-[var(--bg-base)] flex-1 min-h-[calc(100vh-56px)]">
          <GameCenter />
        </div>
        <div className="w-full md:w-1/4 border-t md:border-t-0 md:border-l border-[var(--border-subtle)] bg-[var(--bg-panel)] h-auto md:h-[calc(100vh-56px)] md:sticky md:top-[56px] overflow-hidden">
          <RightPanel />
        </div>
      </div>
      <EducationPanel />
    </div>
  );
}
