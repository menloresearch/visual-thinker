"use client";
import React, { useState, useEffect, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  RotateCcw,
  Send,
  ShuffleIcon,
} from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { useChat } from "@ai-sdk/react";

type MazeData = {
  row: number;
  col: number;
  isCurrentStep: boolean;
  walls: Record<string, boolean>;
  marker: string;
  isPath: boolean;
}[][];

export type MazeStep = MazeData[][][];
export type MazeSample = { Prompt: string; Solution: string };

const MazeSolver = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [maze, setMaze] = useState("");
  const speed = 500;
  const [mazeSteps, setMazeSteps] = useState<MazeData[]>([]);
  const mazeStepsRef = useRef<MazeData[]>([]);
  const currentStepRef = useRef<number>(0);
  const [error, setError] = useState("");
  const [mazePrompts, setMazePrompts] = useState<string[]>([]);

  const {
    messages,
    setInput,
    isLoading,
    handleInputChange,
    handleSubmit,
    setMessages,
  } = useChat();

  const parseMazeState = (mazeText: string): MazeData => {
    try {
      const lines = mazeText
        .trim()
        .split("\n")
        .filter((line) => line.trim().length > 0);
      const mazeData = [];

      for (const line of lines) {
        const tokens =
          line.match(/<\|([^|]+)\|>/g)?.map((token) => token.slice(2, -2)) ||
          [];
        const rowCells = [];

        for (let i = 0; i < tokens.length; i += 3) {
          const [row, col] = tokens[i].split("-").map(Number);
          const wallToken = tokens[i + 1];
          const markerToken = tokens[i + 2];

          const walls: Record<string, boolean> = {};
          if (wallToken !== "no_wall" && wallToken.endsWith("_wall")) {
            const wallDirs = wallToken.slice(0, -5).split("_");
            wallDirs.forEach((dir) => (walls[dir] = true));
          }

          rowCells.push({
            row,
            col,
            walls,
            marker: markerToken,
            isPath:
              ["up", "down", "left", "right", "blank"].includes(markerToken) &&
              markerToken !== "blank",
          });
        }
        mazeData.push(rowCells);
      }
      return mazeData;
    } catch {
      throw new Error("Invalid maze format");
    }
  };

  interface MazeCell {
    row: number;
    col: number;
    walls: Record<string, boolean>;
    marker: string;
    isPath: boolean;
    isCurrentStep?: boolean;
  }

  type MazeData = MazeCell[][];

  const validateMaze = (mazeData: MazeData) => {
    if (!mazeData || !mazeData.length) return false;

    let hasOrigin = false;
    let hasTarget = false;

    for (const row of mazeData) {
      for (const cell of row) {
        if (cell.marker === "origin") hasOrigin = true;
        if (cell.marker === "target") hasTarget = true;
      }
    }

    return hasOrigin && hasTarget;
  };

  const buildPathMap = (steps: MazeData[]) => {
    // Create a map of all cells that are part of the path
    const pathMap = new Map();

    steps.forEach((maze) => {
      maze.forEach((row) => {
        row.forEach((cell) => {
          if (["up", "down", "left", "right"].includes(cell.marker)) {
            const key = `${cell.row}-${cell.col}`;
            pathMap.set(key, true);
          }
        });
      });
    });

    return pathMap;
  };

  const handleInitialMazeInput = (
    event: React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    const input = event.target.value;
    setMaze(input);

    if (!input.length) return;

    try {
      const lines = input.split("\n");
      for (let i = 1; i < lines.length; i++) {
        try {
          const maze = parseMazeState(lines.slice(i).join("\n"));
          if (validateMaze(maze)) {
            mazeStepsRef.current = [maze];
            setMazeSteps([maze]);
            setCurrentStep(0);
            setError("");
          } else {
            setError("Maze must contain both origin (O) and target (T) cells");
            setMazeSteps([]);
          }
          break;
        } catch {
          continue;
        }
      }
    } catch {
      setError("Invalid maze format. Please check your input.");
    }
    handleInputChange(event);
  };

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying && currentStep < mazeSteps.length - 1) {
      timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, speed);
    } else if (currentStep >= mazeSteps.length - 1) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, mazeSteps.length, speed]);

  const togglePlay = () => setIsPlaying(!isPlaying);
  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };
  const nextStep = () =>
    currentStep < mazeSteps.length - 1 && setCurrentStep(currentStep + 1);
  const prevStep = () => currentStep > 0 && setCurrentStep(currentStep - 1);

  const renderMaze = () => {
    if (!mazeStepsRef.current[currentStep]) return null;

    const maze = mazeStepsRef.current[currentStep];
    const maxCols = Math.max(...maze.map((row) => row.length));

    return (
      <div
        className="grid gap-0"
        style={{ gridTemplateColumns: `repeat(${maxCols}, 45px)` }}
      >
        {maze.map((row, rowIdx) =>
          row.map((cell, colIdx) => {
            // Determine cell styling
            let bgColor = "bg-white";
            if (cell.marker === "origin") bgColor = "bg-green-200";
            else if (cell.marker === "target") bgColor = "bg-red-200";
            else if (cell.isCurrentStep) bgColor = "bg-blue-400";
            else if (cell.isPath) bgColor = "bg-blue-200";

            return (
              <div
                key={`${rowIdx}-${colIdx}`}
                className={`w-12 h-12 flex items-center justify-center relative ${bgColor} transition-colors duration-300`}
                style={{
                  margin: "-1px",
                  borderTop: cell.walls.up ? "2px solid black" : undefined,
                  borderBottom: cell.walls.down ? "2px solid black" : undefined,
                  borderLeft: cell.walls.left ? "2px solid black" : undefined,
                  borderRight: cell.walls.right ? "2px solid black" : undefined,
                }}
              >
                {cell.marker === "origin" && "O"}
                {cell.marker === "target" && "T"}
                {(cell.isPath || cell.isCurrentStep) && "•"}
              </div>
            );
          }),
        )}
      </div>
    );
  };

  useEffect(() => {
    try {
      const steps = messages[1]?.content
        .split(/Step \d+:/g)
        .filter((step) => step.trim())
        .map((step) => parseMazeState(step));

      if (steps?.length > currentStepRef.current) {
        // Process steps to show complete path
        const pathMap = buildPathMap(steps);

        const processedSteps = steps.map((maze, stepIndex) => {
          mazeStepsRef.current[stepIndex] = maze;
          return mazeStepsRef.current[0].map((_, rowIndex) => {
            return mazeStepsRef.current[0][rowIndex].map((_, cellIndex) => {
              const cell = maze[rowIndex][cellIndex];

              const key = `${cell.row}-${cell.col}`;
              const isPartOfPath = pathMap.has(key);
              if (isPartOfPath) {
                const isCurrentStep =
                  cell.marker !== "blank" &&
                  cell.marker !== "origin" &&
                  cell.marker !== "target";

                const newData = {
                  ...cell,
                  isPath: isPartOfPath,
                  isCurrentStep,
                };
                mazeStepsRef.current[0][rowIndex][cellIndex] = newData;
                return newData;
              } else {
                return mazeStepsRef.current[0][rowIndex][cellIndex];
              }
            });
          });
        });
        mazeStepsRef.current = processedSteps;
        setMazeSteps([...mazeStepsRef.current]);
        setIsPlaying(false);
        setError("");
      }
    } catch {}
  }, [messages]);

  useEffect(() => {
    if (window)
      setMazePrompts(
        JSON.parse(process.env.NEXT_PUBLIC_PROMPT_SAMPLES ?? "[]")?.map(
          (e: MazeSample) => e.Prompt,
        ) ?? [],
      );
  }, [window]);

  return (
    <div className="w-full h-screen overflow-clip bg-white p-6">
      <CardHeader>
        <CardTitle className="text-center text-3xl font-bold text-blue-600">
          AlphaMAZE
        </CardTitle>
      </CardHeader>

      <div className="flex gap-6 h-[calc(100vh-120px)]">
        {/* Main Maze Display */}
        <div className="flex-1">
          <Card className="h-full bg-white border-2 border-blue-100 shadow-lg">
            <CardContent className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center space-y-6 w-full">
                {/* Maze Visualization */}
                <div className="border-2 border-blue-100 rounded-lg p-6 bg-white h-[70vh] w-full flex items-center justify-center shadow-inner">
                  {renderMaze()}
                </div>

                {/* Playback Controls */}
                <div className="flex items-center space-x-4">
                  <Button
                    variant="outline"
                    onClick={prevStep}
                    disabled={currentStep === 0}
                    className="border-blue-200 hover:bg-blue-50"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    onClick={togglePlay}
                    className="border-blue-200 hover:bg-blue-50"
                  >
                    {isPlaying ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={reset}
                    className="border-blue-200 hover:bg-blue-50"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                  {mazeSteps && (
                    <>
                      <Button
                        variant="outline"
                        onClick={nextStep}
                        disabled={currentStep >= mazeSteps.length - 1}
                        className="border-blue-200 hover:bg-blue-50"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                      <span className="text-sm text-gray-600">
                        Step {currentStep + 1} of {mazeSteps.length || 1}
                      </span>
                    </>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        {/* Right Control Panel */}
        <div className="w-96">
          <Card className="h-full bg-white border-2 border-blue-100 shadow-lg overflow-hidden">
            <CardContent className="space-y-4 p-4">
              {/* Input Area */}
              <div className="relative overflow-hidden">
                <Textarea
                  value={maze}
                  onChange={handleInitialMazeInput}
                  placeholder="Enter your maze configuration..."
                  className="pr-12 h-32 bg-white border-2 border-blue-100 text-gray-800 placeholder:text-gray-400"
                />
                <Button
                  onClick={() => {
                    const randomizedMaze =
                      mazePrompts[
                        Math.floor(Math.random() * mazePrompts.length)
                      ].trim();
                    setMaze(randomizedMaze);
                    setInput(randomizedMaze);
                  }}
                  className="absolute right-2 top-2 bg-blue-600 hover:bg-blue-700 text-white"
                  size="icon"
                >
                  <ShuffleIcon className="h-4 w-4" />
                </Button>
                <Button
                  onClick={() => {
                    setMessages([]);
                    handleSubmit();
                  }}
                  disabled={isLoading || !maze.trim()}
                  className="absolute right-2 top-14 bg-blue-600 hover:bg-blue-700 text-white"
                  size="icon"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>

              {/* Solution Output */}
              <div>
                <h3 className="text-sm font-medium mb-2 text-gray-600">
                  Solution Steps
                </h3>
                <Textarea
                  value={messages[1]?.content}
                  className="w-full h-[calc(100vh-380px)] font-mono text-sm p-2 bg-white border-2 border-blue-100 text-gray-800 rounded"
                  placeholder="Solution steps will appear here..."
                  disabled={true}
                  ref={(el) => {
                    if (el) el.scrollTop = el.scrollHeight;
                  }}
                />
              </div>

              {/* Error Display */}
              {error && (
                <div className="text-red-600 text-sm p-2 bg-red-50 rounded border border-red-200">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default MazeSolver;
