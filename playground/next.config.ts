import type { NextConfig } from "next";
import path from "path";
import fs from "fs";

const jsonlFilePath = path.join(__dirname, "maze_reasoning.jsonl");
const jsonlData = fs
  .readFileSync(jsonlFilePath, "utf-8")
  .split("\n")
  .filter((line) => line.trim() !== "")
  .map((line) => JSON.parse(line));

const nextConfig: NextConfig = {
  /* config options here */
  env: {
    NEXT_PUBLIC_PROMPT_SAMPLES: JSON.stringify(jsonlData),
  },
};

export default nextConfig;
