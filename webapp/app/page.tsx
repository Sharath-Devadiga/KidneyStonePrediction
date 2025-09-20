"use client"; // This is a client-side interactive component

import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Home() {
  // State to hold the form data
  const [formData, setFormData] = useState({
    gravity: 1.020,
    ph: 6.0,
    osmo: 700,
    cond: 21.0,
    urea: 350,
    calc: 4.0,
  });

  // State to hold the prediction result and loading status
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  // Function to handle changes in input fields
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setFormData((prev) => ({ ...prev, [id]: parseFloat(value) }));
  };

  // Function to handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult("");

    try {
      // Send data to our OWN Next.js backend, not the Python API directly
      const response = await axios.post("/api/predict", formData);
      const prediction = response.data.prediction;

      if (prediction === 1) {
        setResult("High Risk of Kidney Stones Detected! 🚨");
      } else {
        setResult("Low Risk of Kidney Stones. ✅");
      }
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setResult("An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-100 p-8">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">
            Kidney Stone Risk Predictor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {Object.entries(formData).map(([key, value]) => (
              <div key={key} className="grid w-full items-center gap-1.5">
                <Label htmlFor={key} className="capitalize">{key}</Label>
                <Input
                  type="number"
                  id={key}
                  value={value}
                  onChange={handleChange}
                  step="any" // Allows decimal points
                  required
                />
              </div>
            ))}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Analyzing..." : "Predict Risk"}
            </Button>
          </form>

          {result && (
            <div className="mt-6 text-center">
              <h3 className="text-lg font-semibold">Prediction Result:</h3>
              <p className={`text-xl font-bold ${result.includes("High Risk") ? "text-red-600" : "text-green-600"}`}>
                {result}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
}