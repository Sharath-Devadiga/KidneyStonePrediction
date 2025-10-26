"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import * as Tabs from '@radix-ui/react-tabs';
import { ImageUpload } from "@/components/ui/image-upload";
import axios from "axios";

// Types
interface UrineData {
  gravity: number | null;
  ph: number | null;
  osmo: number | null;
  cond: number | null;
  urea: number | null;
  calc: number | null;
}

interface PredictionResult {
  prediction: number;
  risk_level: string;
  confidence?: number;
  recommendations: string[];
  timestamp: string;
}

interface HistoryEntry {
  form: UrineData;
  result: PredictionResult;
  time: string;
}

// API call function
const predictKidneyStone = async (data: UrineData): Promise<PredictionResult> => {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error("Prediction failed. Please try again.");
  }
  
  return response.json();
};

// Parameter info with labels and normal ranges
const parameterInfo: Record<keyof UrineData, { label: string; range: string; unit: string }> = {
  gravity: { label: "Specific Gravity", range: "1.005-1.030", unit: "sg" },
  ph: { label: "pH Level", range: "4.5-8.0", unit: "pH" },
  osmo: { label: "Osmolarity", range: "150-1200", unit: "mOsm/kg" },
  cond: { label: "Conductivity", range: "5-35", unit: "mS/cm" },
  urea: { label: "Urea Concentration", range: "50-500", unit: "mg/dL" },
  calc: { label: "Calcium Level", range: "0-15", unit: "mg/dL" },
};

const kidneyFacts = [
  "💧 Kidney stones affect about 1 in 10 people during their lifetime.",
  "🚰 Drinking plenty of water (2-3 liters daily) can help prevent kidney stones.",
  "📏 Some stones can be as small as a grain of sand or as large as a golf ball!",
  "🩺 Symptoms include severe pain, nausea, and blood in urine.",
  "🧬 Diet and genetics both play a role in kidney stone formation.",
  "🥗 A balanced diet low in sodium and animal protein helps prevent stones.",
];

export default function PredictPage() {
  const [form, setForm] = useState<UrineData>({
    gravity: null,
    ph: null,
    osmo: null,
    cond: null,
    urea: null,
    calc: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [factIdx, setFactIdx] = useState(0);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: Number(e.target.value) });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await predictKidneyStone(form);
      setResult(res);
      const entry: HistoryEntry = { 
        form: { ...form }, 
        result: res, 
        time: new Date().toLocaleString() 
      };
      setHistory((prev) => [entry, ...prev].slice(0, 5));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm({ gravity: null, ph: null, osmo: null, cond: null, urea: null, calc: null });
    setResult(null);
    setError(null);
  };

  const exportResults = () => {
    const data = JSON.stringify(history, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kidney-stone-predictions-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setFactIdx((idx) => (idx + 1) % kidneyFacts.length);
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  const [imageLoading, setImageLoading] = useState(false);

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  
  const handleImageUpload = async (file: File) => {
    setImageLoading(true);
    setError(null);
    
    try {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Image = reader.result as string;
        setSelectedImage(base64Image);
        setImageLoading(false);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError("Failed to process image. Please try again.");
      setImageLoading(false);
    }
  };

  const handleAnalyzeImage = async () => {
    if (!selectedImage) return;
    
    setImageLoading(true);
    setError(null);
    
    try {
      const response = await axios.post("/api/predict/image", 
        { image: selectedImage },
      ); 

      const data = response.data;
      setForm(data);
    } catch (err) {
      setError("Failed to analyze image. Please try manual entry.");
    } finally {
      setImageLoading(false);
    }
  };

  // CT/X-Ray scan prediction states and handlers
  const [ctImage, setCtImage] = useState<File | null>(null);
  const [ctPreview, setCtPreview] = useState<string | null>(null);
  const [ctLoading, setCtLoading] = useState(false);
  const [ctResult, setCtResult] = useState<any | null>(null);

  const handleCtImageUpload = async (file: File) => {
    setCtImage(file);
    setCtLoading(false);
    setCtResult(null);
    setError(null);
    
    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setCtPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleCtImagePredict = async () => {
    if (!ctImage) return;
    
    setCtLoading(true);
    setError(null);
    setCtResult(null);
    
    try {
      const formData = new FormData();
      formData.append("file", ctImage);
      
      const response = await axios.post("/api/predict/ct-image", formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });
      
      setCtResult(response.data);
    } catch (err: any) {
      setError(err?.response?.data?.error || "Failed to analyze CT/X-ray image. Please try again.");
    } finally {
      setCtLoading(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50"
    >
      {/* Navigation */}
      <motion.nav 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="sticky top-0 z-50 bg-white/80 backdrop-blur-lg border-b border-slate-200/50 shadow-sm"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link href="/" className="flex items-center space-x-3">
              <motion.div 
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
                className="text-2xl"
              >
                💎
              </motion.div>
              <motion.span 
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="text-xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 bg-clip-text text-transparent"
              >
                Kidney Stone Predictor
              </motion.span>
            </Link>
            <div className="flex items-center gap-3">
              <Link href="/">
                <motion.button 
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-4 py-2 text-sm font-medium text-blue-700 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  🏠 Home
                </motion.button>
              </Link>
              <Link href="/about">
                <motion.button 
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.45 }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-4 py-2 text-sm font-medium text-blue-700 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  ℹ️ About
                </motion.button>
              </Link>
            </div>
          </div>
        </div>
      </motion.nav>

      <motion.main 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8"
      >
        <div className="grid lg:grid-cols-12 gap-8">
          {/* Main Content */}
          <motion.div 
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="lg:col-span-8 space-y-6"
          >
            {/* Prediction Form Card */}
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200/50 overflow-hidden">
              <motion.div
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ type: "spring", stiffness: 100, delay: 0.5 }}
                className="border-b border-slate-200/50 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 px-6 py-5"
              >
                <div className="flex items-center gap-3">
                  <motion.span 
                    animate={{ rotate: [0, -10, 10, -10, 10, 0] }}
                    transition={{ duration: 1, delay: 1 }}
                    className="text-3xl"
                  >
                    🔬
                  </motion.span>
                  <div>
                    <h2 className="text-2xl font-bold text-slate-800">Urine Analysis Parameters</h2>
                    <p className="text-sm text-slate-600 mt-1">Upload an image or enter your test results for AI-powered prediction</p>
                  </div>
                </div>
              </motion.div>

              <div className="p-6">
                <Tabs.Root defaultValue="manual">
                  <Tabs.List className="flex space-x-2 border-b border-slate-200 mb-6">
                    <Tabs.Trigger
                      value="manual"
                      className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 data-[state=active]:text-blue-600 data-[state=active]:border-b-2 data-[state=active]:border-blue-600"
                    >
                      ⌨️ Manual Entry
                    </Tabs.Trigger>
                    <Tabs.Trigger
                      value="image"
                      className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 data-[state=active]:text-blue-600 data-[state=active]:border-b-2 data-[state=active]:border-blue-600"
                    >
                      � Urine Test Image
                    </Tabs.Trigger>
                    <Tabs.Trigger
                      value="ct-scan"
                      className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 data-[state=active]:text-blue-600 data-[state=active]:border-b-2 data-[state=active]:border-blue-600"
                    >
                      🏥 CT/X-Ray Scan
                    </Tabs.Trigger>
                  </Tabs.List>

                  <Tabs.Content value="manual" className="space-y-6">
                    <form onSubmit={handleSubmit} className="space-y-8">
                      <div className="grid sm:grid-cols-2 gap-6">
                        {(Object.entries(form) as [keyof UrineData, number][]).map(([key, value], index) => (
                          <motion.div 
                            key={key}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.6 + index * 0.05 }}
                            className="space-y-2"
                          >
                            <label 
                              htmlFor={key} 
                              className="block text-sm font-semibold text-slate-700"
                            >
                              {parameterInfo[key].label}
                              <span className="ml-2 text-xs font-normal text-slate-500">
                                ({parameterInfo[key].range} {parameterInfo[key].unit})
                              </span>
                            </label>
                            <input
                              type="number"
                              id={key}
                              name={key}
                              value={value ?? ""}
                              onChange={handleChange}
                              step="any"
                              required
                              className="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all outline-none"
                              placeholder={`Enter ${parameterInfo[key].label.toLowerCase()}`}
                            />
                          </motion.div>
                        ))}
                      </div>

                      <div className="flex gap-4">
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          type="submit" 
                          className="flex-1 py-3.5 text-base font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed" 
                          disabled={loading}
                        >
                          {loading ? (
                            <span className="flex items-center gap-3 justify-center">
                              <motion.span 
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="inline-block w-5 h-5 border-3 border-white/30 border-t-white rounded-full"
                              />
                              <span>Analyzing Results...</span>
                            </span>
                          ) : (
                            <span className="flex items-center gap-2 justify-center">
                              <span>🔍</span>
                              <span>Predict Risk</span>
                            </span>
                          )}
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          type="button" 
                          className="px-8 py-3.5 border-2 border-slate-300 rounded-lg hover:bg-slate-50 transition-all font-medium text-slate-700" 
                          onClick={handleReset}
                        >
                          Reset
                        </motion.button>
                      </div>
                    </form>
                  </Tabs.Content>

                  <Tabs.Content value="image" className="space-y-6">
                    <div className="space-y-4">
                      <ImageUpload onImageUpload={handleImageUpload} isLoading={imageLoading} />
                      {selectedImage && (
                        <div className="mt-4 flex justify-center">
                          <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleAnalyzeImage}
                            className="py-2.5 px-6 text-sm font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={imageLoading}
                          >
                            {imageLoading ? (
                              <span className="flex items-center gap-3 justify-center">
                                <motion.span 
                                  animate={{ rotate: 360 }}
                                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                  className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                                />
                                <span>Analyzing Image...</span>
                              </span>
                            ) : (
                              <span className="flex items-center gap-2 justify-center">
                                <span>🔬</span>
                                <span>Analyze Image</span>
                              </span>
                            )}
                          </motion.button>
                        </div>
                      )}
                      {Object.entries(form).some(([, value]) => value !== null) && (
                        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                          <h3 className="text-sm font-semibold text-blue-700 mb-2">Detected Parameters:</h3>
                          <div className="grid sm:grid-cols-2 gap-4">
                            {Object.entries(form).map(([key, value]) => (
                              <div key={key} className="bg-white p-2 rounded shadow-sm">
                                <span className="text-xs text-slate-600">{parameterInfo[key as keyof UrineData].label}:</span>
                                <span className="ml-2 text-sm font-medium">{value ?? "Not detected"}</span>
                              </div>
                            ))}
                          </div>
                          <div className="mt-4 flex gap-4">
                            <motion.button
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                              onClick={handleSubmit}
                              className="flex-1 py-2.5 text-sm font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                              disabled={loading}
                            >
                              Predict with Detected Values
                            </motion.button>
                            <motion.button
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                              onClick={handleReset}
                              className="px-4 py-2.5 text-sm border-2 border-slate-300 rounded-lg hover:bg-slate-50 transition-all font-medium text-slate-700"
                            >
                              Reset
                            </motion.button>
                          </div>
                        </div>
                      )}
                    </div>
                  </Tabs.Content>

                  <Tabs.Content value="ct-scan" className="space-y-6">
                    <div className="space-y-4">
                      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                        <p className="text-sm text-blue-800">
                          <strong>🏥 CT/X-Ray Scan Analysis:</strong> Upload a kidney CT scan or X-ray image for AI-powered stone detection.
                        </p>
                      </div>
                      
                      <ImageUpload onImageUpload={handleCtImageUpload} isLoading={ctLoading} />
                      
                      {ctPreview && (
                        <div className="mt-4 flex justify-center">
                          <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleCtImagePredict}
                            className="py-3 px-8 text-sm font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={ctLoading}
                          >
                            {ctLoading ? (
                              <span className="flex items-center gap-3 justify-center">
                                <motion.span 
                                  animate={{ rotate: 360 }}
                                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                  className="inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                                />
                                <span>Analyzing Scan...</span>
                              </span>
                            ) : (
                              <span className="flex items-center gap-2 justify-center">
                                <span>🔬</span>
                                <span>Analyze CT/X-Ray Image</span>
                              </span>
                            )}
                          </motion.button>
                        </div>
                      )}
                      
                      {ctResult && (
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-6 p-6 bg-white border-2 border-blue-200 rounded-xl shadow-lg"
                        >
                          <div className="flex items-center gap-3 mb-4">
                            <div className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${
                              ctResult.classification === 'Stone' 
                                ? 'bg-red-100' 
                                : 'bg-green-100'
                            }`}>
                              {ctResult.classification === 'Stone' ? '🔴' : '✅'}
                            </div>
                            <div>
                              <h3 className="text-xl font-bold text-slate-800">
                                {ctResult.classification === 'Stone' ? 'Kidney Stone Detected' : 'No Stone Detected'}
                              </h3>
                              <p className="text-sm text-slate-600">
                                Confidence: {(ctResult.confidence * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4 mb-4">
                            <div className="p-3 bg-slate-50 rounded-lg">
                              <p className="text-xs text-slate-600 mb-1">Probability Normal</p>
                              <p className="text-lg font-semibold text-green-600">
                                {(ctResult.probability_normal * 100).toFixed(1)}%
                              </p>
                            </div>
                            <div className="p-3 bg-slate-50 rounded-lg">
                              <p className="text-xs text-slate-600 mb-1">Probability Stone</p>
                              <p className="text-lg font-semibold text-red-600">
                                {(ctResult.probability_stone * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                          
                          <div className="border-t border-slate-200 pt-4">
                            <h4 className="font-semibold text-slate-800 mb-3">📋 Recommendations:</h4>
                            <ul className="space-y-2">
                              {ctResult.recommendations.map((rec: string, idx: number) => (
                                <motion.li
                                  key={idx}
                                  initial={{ opacity: 0, x: -10 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: idx * 0.1 }}
                                  className="flex items-start gap-2 text-sm text-slate-700"
                                >
                                  <span className="text-blue-500 mt-0.5">•</span>
                                  <span>{rec}</span>
                                </motion.li>
                              ))}
                            </ul>
                          </div>
                          
                          <div className="mt-4 text-xs text-slate-500">
                            <p>Model: {ctResult.model_type}</p>
                            <p>Analyzed: {new Date(ctResult.timestamp).toLocaleString()}</p>
                          </div>
                        </motion.div>
                      )}
                    </div>
                  </Tabs.Content>
                </Tabs.Root>
                
                <AnimatePresence mode="wait">
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg"
                    >
                      <div className="flex items-center gap-2 text-red-700">
                        <span className="text-xl">⚠️</span>
                        <p className="font-medium">{error}</p>
                      </div>
                    </motion.div>
                  )}
                  
                  {result !== null && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.9, y: 20 }}
                      transition={{ type: "spring", stiffness: 200 }}
                      className={`mt-6 p-6 rounded-xl shadow-lg ${
                        result.prediction === 1 
                          ? "bg-gradient-to-br from-red-50 to-red-100 border-2 border-red-200" 
                          : "bg-gradient-to-br from-green-50 to-emerald-100 border-2 border-green-200"
                      }`}
                    >
                      <div className="flex flex-col items-center text-center">
                        <motion.div 
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: "spring", stiffness: 200, delay: 0.1 }}
                          className="text-5xl mb-4"
                        >
                          {result.prediction === 1 ? "⚠️" : "✅"}
                        </motion.div>
                        <h3 className={`text-2xl font-bold mb-2 ${
                          result.prediction === 1 ? "text-red-700" : "text-green-700"
                        }`}>
                          {result.risk_level}
                        </h3>
                        {result.confidence && (
                          <p className="text-sm text-slate-600 mb-4">
                            Confidence: {(result.confidence * 100).toFixed(1)}%
                          </p>
                        )}
                        
                        <div className="w-full mt-4 p-4 bg-white/50 rounded-lg text-left">
                          <h4 className="font-semibold text-slate-800 mb-2">📋 Recommendations:</h4>
                          <ul className="space-y-2">
                            {result.recommendations.map((rec, idx) => (
                              <motion.li 
                                key={idx}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.2 + idx * 0.1 }}
                                className="text-sm text-slate-700 flex items-start gap-2"
                              >
                                <span className="mt-0.5">•</span>
                                <span>{rec}</span>
                              </motion.li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
                
                {history.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="mt-8 bg-slate-50/50 rounded-xl border border-slate-200/50 p-6"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                        <span className="text-xl">📜</span>
                        Recent Predictions ({history.length})
                      </h3>
                      <div className="flex gap-2">
                        <button 
                          className="text-sm text-green-600 hover:text-green-700 font-medium px-3 py-1 border border-green-300 rounded-lg hover:bg-green-50 transition-colors" 
                          onClick={exportResults}
                        >
                          📥 Export
                        </button>
                        <button 
                          className="text-sm text-red-600 hover:text-red-700 font-medium px-3 py-1 border border-red-300 rounded-lg hover:bg-red-50 transition-colors" 
                          onClick={() => setHistory([])}
                        >
                          🗑️ Clear
                        </button>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {history.map((h, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="bg-white rounded-lg p-4 border border-slate-200 hover:shadow-md transition-shadow"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-slate-500">{h.time}</span>
                            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
                              h.result.prediction === 1 
                                ? "bg-red-100 text-red-700" 
                                : "bg-green-100 text-green-700"
                            }`}>
                              {h.result.prediction === 1 ? "🚨 High Risk" : "✅ Low Risk"}
                              {h.result.confidence && ` (${(h.result.confidence * 100).toFixed(0)}%)`}
                            </span>
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-xs">
                            {(Object.entries(h.form) as [keyof UrineData, number][]).map(([k, v]) => (
                              <div key={k} className="bg-slate-50 px-2 py-1 rounded">
                                <span className="font-medium text-slate-600">{parameterInfo[k].label}:</span>
                                <span className="ml-1 text-slate-800">{v}</span>
                              </div>
                            ))}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
          
          {/* Sidebar */}
          <div className="lg:col-span-4 space-y-6">
            {/* Educational Facts Card */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="bg-white rounded-2xl shadow-xl border border-slate-200/50 overflow-hidden"
            >
              <div className="border-b border-slate-200/50 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 px-6 py-4">
                <div className="flex items-center gap-3">
                  <motion.span
                    animate={{ 
                      scale: [1, 1.2, 1],
                      rotate: [0, 10, -10, 0]
                    }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      repeatDelay: 5
                    }}
                    className="text-2xl"
                  >
                    💡
                  </motion.span>
                  <h3 className="text-xl font-bold text-slate-800">Did You Know?</h3>
                </div>
              </div>
              <div className="p-6">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={factIdx}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.5 }}
                    className="text-slate-700 leading-relaxed p-4 bg-blue-50/50 rounded-lg border border-blue-100/50"
                  >
                    {kidneyFacts[factIdx]}
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.div>
            
            {/* Parameter Guide Card */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="bg-white rounded-2xl shadow-xl border border-slate-200/50 overflow-hidden"
            >
              <div className="border-b border-slate-200/50 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 px-6 py-4">
                <div className="flex items-center gap-3">
                  <motion.span
                    animate={{ 
                      rotate: [0, -10, 10, -10, 0],
                      scale: [1, 1.1, 1]
                    }}
                    transition={{ 
                      duration: 1.5,
                      repeat: Infinity,
                      repeatDelay: 5
                    }}
                    className="text-2xl"
                  >
                    📖
                  </motion.span>
                  <h3 className="text-xl font-bold text-slate-800">Parameter Guide</h3>
                </div>
              </div>
              <div className="p-6">
                <div className="space-y-3">
                  {(Object.entries(parameterInfo) as [keyof UrineData, typeof parameterInfo[keyof UrineData]][]).map(([key, info], index) => (
                    <motion.div
                      key={key}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.7 + index * 0.05 }}
                      whileHover={{ scale: 1.02, x: 5 }}
                      className="p-3 rounded-lg bg-slate-50 hover:bg-blue-50/50 transition-all duration-200 border border-slate-200/50 cursor-pointer"
                    >
                      <div className="font-semibold text-slate-800 mb-1">{info.label}</div>
                      <div className="flex items-center gap-2 text-sm text-slate-600">
                        <span className="text-blue-500">📊</span>
                        <span>
                          Normal: <span className="font-medium text-blue-600">{info.range} {info.unit}</span>
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.main>
    </motion.div>
  );
}