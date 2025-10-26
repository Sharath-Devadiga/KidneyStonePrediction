import { NextResponse } from "next/server";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error("Failed to fetch health status");
    }
    
    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error checking health:", error);
    return NextResponse.json(
      { 
        status: "unhealthy",
        error: "Backend is not reachable",
        message: "Make sure FastAPI server is running on port 8000"
      },
      { status: 503 }
    );
  }
}
