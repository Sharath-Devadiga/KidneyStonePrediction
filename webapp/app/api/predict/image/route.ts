import { NextRequest, NextResponse } from "next/server";
import { openai } from "@/lib/server/openai";

export async function POST(req: NextRequest) {
    try {
        // You should re-enable session validation in production
        // if (!session || !session.user) {
        //     return NextResponse.json(
        //         { msg: "You are not authorised to access this endpoint" },
        //         { status: 401 }
        //     );
        // }
        console.log("hi")
        const body = await req.json();
        const { image } = body;

        // Manual validation
        if (!image || typeof image !== "string" || !image.startsWith("data:image/")) {
            return NextResponse.json(
                { msg: "Invalid Inputs. Expected a base64 image string." },
                { status: 400 }
            );
        }

        const base64Data = image.split(',')[1];
        const mimeType = image.match(/data:(image\/[a-z]+);/)?.[1] || "image/jpeg";

        const systemPrompt = `You are a world-class medical AI specializing in nephrology. Your task is to analyze a urine test report image and predict the risk of kidney stones.

You must respond in a specific JSON format. Do not include any other text, explanations, or markdown wrappers.

Based on the parameters in the image (like specific gravity, pH, blood, leukocytes, etc.), determine the risk level.

The JSON output must have the following structure:
{
  "prediction": <0 for Low Risk, 1 for High Risk>,
  "risk_level": <"Low Risk" or "High Risk">,
  "confidence": <A number between 0.0 and 1.0 representing your confidence>,
  "recommendations": [
    "<Recommendation 1>",
    "<Recommendation 2>",
    "..."
  ]
}

Example for High Risk:
{
  "prediction": 1,
  "risk_level": "High Risk",
  "confidence": 0.85,
  "recommendations": [
    "High pH and specific gravity indicate a risk for stone formation.",
    "Increase daily water intake to 2-3 liters.",
    "Consult a nephrologist or urologist for further evaluation.",
    "Consider dietary changes, such as reducing sodium and animal protein."
  ]
}

Example for Low Risk:
{
  "prediction": 0,
  "risk_level": "Low Risk",
  "confidence": 0.95,
  "recommendations": [
    "All parameters appear to be within the normal range.",
    "Continue to maintain a healthy diet and adequate hydration.",
    "No immediate concerns for kidney stones are indicated by this report."
  ]
}`;

        const response = await openai.chat.completions.create({
            model: "gemini-2.5-flash",
            messages: [
                {
                    role: "system",
                    content: systemPrompt
                },
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: "Analyze this urine test report and provide the risk prediction in the required JSON format."
                        },
                        {
                            type: "image_url",
                            image_url: {
                                url: `data:${mimeType};base64,${base64Data}`
                            }
                        }
                    ]
                }
            ],
            temperature: 0.1,
            max_tokens: 1024
        });

        const messageContent = response.choices[0]?.message?.content;
        if (!messageContent) {
            throw new Error("No response content from AI");
        }

        console.log("AI Response:", messageContent);

        try {
            // Clean up potential markdown wrappers
            let jsonText = messageContent.trim();
            if (jsonText.startsWith("```json")) {
                jsonText = jsonText.substring(7, jsonText.length - 3).trim();
            } else if (jsonText.startsWith("```")) {
                jsonText = jsonText.substring(3, jsonText.length - 3).trim();
            }

            const parsedJson = JSON.parse(jsonText);
            
            // Add a timestamp to match the PredictionResult interface
            const finalResult = {
                ...parsedJson,
                timestamp: new Date().toISOString()
            };

            return NextResponse.json(finalResult, { status: 200 });

        } catch (parseError) {
            console.error("Failed to parse AI response as JSON:", messageContent);
            return NextResponse.json(
                { 
                    msg: "Could not parse AI response",
                    rawResponse: messageContent 
                },
                { status: 422 }
            );
        }

    } catch (error: any) {
        console.error("Error analyzing image:", error.message);
        return NextResponse.json(
            { 
                msg: "Failed to analyze image. Please ensure the image is clear and contains a urine test report.",
                error: error.message 
            },
            { status: 500 }
        );
    }
}