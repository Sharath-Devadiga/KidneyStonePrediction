"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import { useState, useEffect } from "react";

export default function AboutPage() {
  const [apiStats, setApiStats] = useState<any>(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/statistics")
      .then((res) => res.json())
      .then((data) => setApiStats(data))
      .catch(() => console.log("Could not load API stats"));
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50"
    >
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-lg border-b border-slate-200/50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link href="/" className="flex items-center space-x-3">
              <span className="text-2xl">💎</span>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 bg-clip-text text-transparent">
                Kidney Stone Predictor
              </span>
            </Link>
            <div className="flex items-center gap-3">
              <Link href="/">
                <button className="px-4 py-2 text-sm font-medium text-blue-700 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors">
                  🏠 Home
                </button>
              </Link>
              <Link href="/predict">
                <button className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
                  🔬 Predict
                </button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            About This Project
          </h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            AI-powered kidney stone risk prediction using advanced machine learning
          </p>
        </motion.div>

        <div className="space-y-6">
          {/* What is this section */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-2xl shadow-lg border border-slate-200/50 p-8"
          >
            <h2 className="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-3">
              <span className="text-3xl">🎯</span>
              What is This?
            </h2>
            <p className="text-slate-700 leading-relaxed mb-4">
              This is a professional-grade machine learning application that predicts kidney stone risk based on urine analysis parameters. 
              The system uses a Random Forest Classifier trained on real medical data to provide accurate, instant predictions.
            </p>
            <p className="text-slate-700 leading-relaxed">
              The application combines modern web technologies (Next.js, React, TypeScript) with a robust Python FastAPI backend to deliver 
              a seamless, production-ready experience.
            </p>
          </motion.div>

          {/* How it works */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-2xl shadow-lg border border-slate-200/50 p-8"
          >
            <h2 className="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-3">
              <span className="text-3xl">⚙️</span>
              How It Works
            </h2>
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">1</div>
                <div>
                  <h3 className="font-semibold text-slate-800 mb-1">Data Collection</h3>
                  <p className="text-slate-600 text-sm">User inputs six key urine analysis parameters: specific gravity, pH, osmolarity, conductivity, urea, and calcium levels.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">2</div>
                <div>
                  <h3 className="font-semibold text-slate-800 mb-1">API Processing</h3>
                  <p className="text-slate-600 text-sm">The Next.js API route receives the data and forwards it to the FastAPI backend running on port 8000.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">3</div>
                <div>
                  <h3 className="font-semibold text-slate-800 mb-1">ML Prediction</h3>
                  <p className="text-slate-600 text-sm">The trained Random Forest model analyzes the parameters and generates a prediction with confidence scores.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">4</div>
                <div>
                  <h3 className="font-semibold text-slate-800 mb-1">Results & Recommendations</h3>
                  <p className="text-slate-600 text-sm">The system returns a risk assessment along with personalized health recommendations based on your specific parameters.</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Tech Stack */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="bg-white rounded-2xl shadow-lg border border-slate-200/50 p-8"
          >
            <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-3">
              <span className="text-3xl">🛠️</span>
              Technology Stack
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-slate-800 mb-3 text-lg">Frontend</h3>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Next.js 15 (React 19)</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>TypeScript</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Tailwind CSS</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Framer Motion (animations)</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Radix UI Components</span>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-slate-800 mb-3 text-lg">Backend</h3>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Python FastAPI</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Scikit-learn (Random Forest)</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Pandas & NumPy</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Pydantic (data validation)</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-600">
                    <span className="text-blue-500">▸</span>
                    <span>Joblib (model serialization)</span>
                  </li>
                </ul>
              </div>
            </div>
          </motion.div>

          {/* Model Information */}
          {apiStats && apiStats.features && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="bg-white rounded-2xl shadow-lg border border-slate-200/50 p-8"
            >
              <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-3">
                <span className="text-3xl">🤖</span>
                Model Information
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                    <span className="font-medium text-slate-700">Model Type:</span>
                    <span className="text-blue-600 font-semibold">{apiStats.model_type}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                    <span className="font-medium text-slate-700">Features:</span>
                    <span className="text-blue-600 font-semibold">{apiStats.feature_count}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                    <span className="font-medium text-slate-700">Version:</span>
                    <span className="text-blue-600 font-semibold">{apiStats.version}</span>
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold text-slate-800 mb-3">Input Features:</h3>
                  <div className="space-y-2">
                    {apiStats.features.map((feature: string, idx: number) => (
                      <div key={idx} className="flex items-center gap-2 text-sm text-slate-600">
                        <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                        <span className="capitalize">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Model Error State */}
          {!apiStats && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="bg-yellow-50 border-2 border-yellow-200 rounded-2xl p-8"
            >
              <h2 className="text-2xl font-bold text-yellow-800 mb-4 flex items-center gap-3">
                <span className="text-3xl">⚠️</span>
                Model Status
              </h2>
              <p className="text-slate-700 leading-relaxed mb-3">
                The ML model is currently experiencing compatibility issues. This is likely due to a scikit-learn version mismatch.
              </p>
              <div className="bg-white rounded-lg p-4 mb-3">
                <h3 className="font-semibold text-slate-800 mb-2">📋 Model Details:</h3>
                <ul className="space-y-1 text-sm text-slate-600">
                  <li>• <strong>Model Type:</strong> Random Forest Classifier</li>
                  <li>• <strong>Features:</strong> 6 (gravity, ph, osmo, cond, urea, calc)</li>
                  <li>• <strong>Version:</strong> 2.0.0</li>
                </ul>
              </div>
              <p className="text-sm text-slate-600">
                To fix: Please retrain the model using the current scikit-learn version by running the Jupyter notebook in <code className="bg-slate-200 px-2 py-1 rounded">notebooks/</code>
              </p>
            </motion.div>
          )}

          {/* Features */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7 }}
            className="bg-white rounded-2xl shadow-lg border border-slate-200/50 p-8"
          >
            <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-3">
              <span className="text-3xl">✨</span>
              Key Features
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100/50 rounded-lg border border-blue-200/50">
                <h3 className="font-semibold text-blue-800 mb-2">🚀 Real-time Predictions</h3>
                <p className="text-sm text-slate-600">Instant analysis with sub-second response times</p>
              </div>
              <div className="p-4 bg-gradient-to-br from-green-50 to-green-100/50 rounded-lg border border-green-200/50">
                <h3 className="font-semibold text-green-800 mb-2">📊 Confidence Scores</h3>
                <p className="text-sm text-slate-600">Probability-based predictions for transparency</p>
              </div>
              <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100/50 rounded-lg border border-purple-200/50">
                <h3 className="font-semibold text-purple-800 mb-2">💡 Smart Recommendations</h3>
                <p className="text-sm text-slate-600">Personalized health advice based on your results</p>
              </div>
              <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100/50 rounded-lg border border-orange-200/50">
                <h3 className="font-semibold text-orange-800 mb-2">📜 History Tracking</h3>
                <p className="text-sm text-slate-600">Monitor trends with prediction history</p>
              </div>
              <div className="p-4 bg-gradient-to-br from-pink-50 to-pink-100/50 rounded-lg border border-pink-200/50">
                <h3 className="font-semibold text-pink-800 mb-2">📥 Export Functionality</h3>
                <p className="text-sm text-slate-600">Download your prediction data as JSON</p>
              </div>
              <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100/50 rounded-lg border border-indigo-200/50">
                <h3 className="font-semibold text-indigo-800 mb-2">🎨 Modern UI/UX</h3>
                <p className="text-sm text-slate-600">Beautiful, responsive design with animations</p>
              </div>
            </div>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="bg-yellow-50 border-2 border-yellow-200 rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold text-yellow-800 mb-4 flex items-center gap-3">
              <span className="text-3xl">⚠️</span>
              Important Disclaimer
            </h2>
            <p className="text-slate-700 leading-relaxed mb-3">
              This application is intended for <strong>educational and informational purposes only</strong>. It is NOT a substitute for 
              professional medical advice, diagnosis, or treatment.
            </p>
            <p className="text-slate-700 leading-relaxed">
              Always consult with qualified healthcare professionals regarding any medical concerns. The predictions made by this system 
              should be used as a supplementary tool, not as a definitive medical diagnosis.
            </p>
          </motion.div>

          {/* CTA */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.9 }}
            className="text-center pt-8"
          >
            <Link href="/predict">
              <button className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all text-lg">
                Try Prediction Now →
              </button>
            </Link>
          </motion.div>
        </div>
      </main>
    </motion.div>
  );
}
