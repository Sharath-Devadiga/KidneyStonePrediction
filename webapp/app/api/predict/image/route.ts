import { NextRequest, NextResponse } from "next/server";
import { openai } from "@/lib/server/openai";

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { image } = body;

        // Manual validation
        if (!image || typeof image !== "string" || !image.startsWith("data:image/")) {
            return NextResponse.json(
                { msg: "Invalid Inputs. Expected a base64 image string." },
                { status: 400 }
            );
        }

        // Convert data URI to just base64 if needed
        // Gemini might not accept the data:image/jpeg;base64, prefix
        const base64Data = image.includes(',') ? image.split(',')[1] : image;
        const mimeType = image.match(/data:(image\/[a-z]+);/)?.[1] || "image/jpeg";

        const userPrompt = `Analyze this urine test result image. Extract and return ONLY these parameters in JSON format:
{
  "gravity": (specific gravity value as number),
  "ph": (pH value as number),
  "osmo": (osmolarity value as number),
  "cond": (conductivity value as number),
  "urea": (urea concentration value as number),
  "calc": (calcium level value as number)
}

If any value cannot be determined from the image, use null for that field. 

IMPORTANT: Return ONLY the JSON object with no additional text, explanations, or markdown formatting.`;

        // Try the Gemini-specific format
        const response = await openai.chat.completions.create({
            model: "gemini-2.0-flash-exp",
            messages: [
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: userPrompt
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
            max_tokens: 500
        });

        const messageContent = response.choices[0]?.message?.content;
        if (!messageContent) {
            throw new Error("No response content from AI");
        }

        console.log("AI Response:", messageContent);

        try {
            // Try to parse the response as JSON first
            const parsedJson = JSON.parse(messageContent);
            
            // Validate the response has the expected structure
            const validatedData = {
                gravity: parsedJson.gravity !== undefined ? parsedJson.gravity : null,
                ph: parsedJson.ph !== undefined ? parsedJson.ph : null,
                osmo: parsedJson.osmo !== undefined ? parsedJson.osmo : null,
                cond: parsedJson.cond !== undefined ? parsedJson.cond : null,
                urea: parsedJson.urea !== undefined ? parsedJson.urea : null,
                calc: parsedJson.calc !== undefined ? parsedJson.calc : null,
            };
            
            return NextResponse.json(validatedData, { status: 200 });
        } catch (parseError) {
            // If direct parsing fails, try to extract JSON from markdown
            try {
                let jsonText = messageContent.trim();
                
                // Remove markdown code blocks if present
                if (jsonText.includes("```")) {
                    const matches = jsonText.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
                    if (matches && matches[1]) {
                        jsonText = matches[1];
                    }
                }
                
                const cleanedJson = JSON.parse(jsonText);
                
                const validatedData = {
                    gravity: cleanedJson.gravity !== undefined ? cleanedJson.gravity : null,
                    ph: cleanedJson.ph !== undefined ? cleanedJson.ph : null,
                    osmo: cleanedJson.osmo !== undefined ? cleanedJson.osmo : null,
                    cond: cleanedJson.cond !== undefined ? cleanedJson.cond : null,
                    urea: cleanedJson.urea !== undefined ? cleanedJson.urea : null,
                    calc: cleanedJson.calc !== undefined ? cleanedJson.calc : null,
                };
                
                return NextResponse.json(validatedData, { status: 200 });
            } catch (secondaryError) {
                console.error("Failed to parse AI response as JSON:", messageContent);
                return NextResponse.json(
                    { 
                        msg: "Could not parse AI response",
                        rawResponse: messageContent 
                    },
                    { status: 422 }
                );
            }
        }

    } catch (error: any) {
        console.error("Error analyzing image:");
        console.error("Error message:", error.message);
        console.error("Error stack:", error.stack);
        if (error.response) {
            console.error("Response data:", error.response.data);
            console.error("Response status:", error.response.status);
        }
        
        return NextResponse.json(
            { 
                msg: "Failed to analyze image. Please ensure the image is clear and contains a urine test report.",
                error: error.message 
            },
            { status: 500 }
        );
    }
}