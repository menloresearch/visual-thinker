import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
// This method must be named GET
export async function POST(req: Request) {
  // Make a request to OpenAI's API based on
  // a placeholder prompt
  const { messages } = await req.json();
  const DEFAULT_MODEL = "jan-hq/AlphaMaze-v0.2-1.5B-GRPO-cp-600";
  const modelName = process.env.MODEL_NAME || DEFAULT_MODEL;
  
  const openai = createOpenAI({
    baseURL: process.env.API_URL,
    apiKey: "menlo",
  });
  const response = streamText({
    model: openai(modelName),
    temperature: 0.6,
    messages,
  });
  return response.toDataStreamResponse({
    headers: {
      "Content-Type": "text/event-stream",
    },
  });
}
