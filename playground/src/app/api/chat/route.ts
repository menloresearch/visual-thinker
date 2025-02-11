import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
// This method must be named GET
export async function POST(req: Request) {
  // Make a request to OpenAI's API based on
  // a placeholder prompt
  const { messages } = await req.json();

  const openai = createOpenAI({
    baseURL: process.env.API_URL,
    apiKey: "menlo",
  });
  const response = streamText({
    model: openai("./Deepseek-Qwen2.5-7B-Redistil-GRPO"),
    messages,
  });
  return response.toDataStreamResponse({
    headers: {
      "Content-Type": "text/event-stream",
    },
  });
}
