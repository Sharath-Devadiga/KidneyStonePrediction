import axios from "axios";
import { NextResponse } from "next/server";

// This is the backend endpoint that the frontend will call.
export async function POST(request: Request) {
  try {
    // 1. Get the urine data from the frontend's request
    const body = await request.json();

    // 2. FORWARD this data to our Python FastAPI server
    //    This is server-to-server communication.
    const pythonApiUrl = "http://127.0.0.1:8000/predict";
    const apiResponse = await axios.post(pythonApiUrl, body);
    
    // 3. Get the prediction from the Python API's response
    const prediction = apiResponse.data;

    // 4. Send the final prediction back to the frontend
    return NextResponse.json(prediction, { status: 200 });

  } catch (error) {
    console.error("Error in Next.js backend:", error);
    return NextResponse.json(
      { error: "Failed to get prediction from model" },
      { status: 500 }
    );
  }
}