import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
    try {
        const formData = await req.formData();
        const file = formData.get("file") as File;

        if (!file) {
            return NextResponse.json(
                { error: "No file uploaded" },
                { status: 400 }
            );
        }

        // Validate file type
        if (!file.type.startsWith("image/")) {
            return NextResponse.json(
                { error: "File must be an image" },
                { status: 400 }
            );
        }

        // Forward the request to the FastAPI backend
        const backendFormData = new FormData();
        backendFormData.append("file", file);

        const response = await fetch(`${API_URL}/predict/image`, {
            method: "POST",
            body: backendFormData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            return NextResponse.json(
                { error: errorData.detail || "Prediction failed" },
                { status: response.status }
            );
        }

        const result = await response.json();
        return NextResponse.json(result, { status: 200 });

    } catch (error) {
        console.error("Image prediction error:", error);
        return NextResponse.json(
            { error: "Internal server error during image prediction" },
            { status: 500 }
        );
    }
}
