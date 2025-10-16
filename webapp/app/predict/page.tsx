"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Mock prediction function (replace with your actual API call)
const predictKidneyStone = async (data) => {
  await new Promise(resolve => setTimeout(resolve, 1500));
  const riskScore = (data.gravity + data.ph + data.calc) / 3;
  return { prediction: riskScore > 5 ? 1 : 0 };
};

// Parameter info with labels and normal ranges
const parameterInfo = {
  gravity: { label: "Specific Gravity", range: "1.005-1.030" },
  ph: { label: "pH Level", range: "4.5-8.0" },
  osmo: { label: "Osmolarity", range: "150-1200 mOsm/kg" },
  cond: { label: "Conductivity", range: "5-35 mS/cm" },
  urea: { label: "Urea Concentration", range: "50-500 mg/dL" },
  calc: { label: "Calcium Level", range: "0-15 mg/dL" },
};

const kidneyFacts = [
  "Kidney stones affect about 1 in 10 people during their lifetime.",
  "Drinking plenty of water can help prevent kidney stones.",
  "Some stones can be as small as a grain of sand or as large as a golf ball!",
  "Symptoms include severe pain, nausea, and blood in urine.",
  "Diet and genetics both play a role in kidney stone formation.",
];

export default function PredictPage() {
  const [form, setForm] = useState({
    gravity: 1.015,
    ph: 6.0,
    osmo: 400,
    cond: 15.0,
    urea: 150,
    calc: 2.0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [factIdx, setFactIdx] = useState(0);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: Number(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictKidneyStone(form);
      setResult(res.prediction);
      const entry = { form, result: res.prediction, time: new Date().toLocaleString() };
      setHistory((prev) => [entry, ...prev].slice(0, 5));
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm({ gravity: 1.015, ph: 6.0, osmo: 400, cond: 15.0, urea: 150, calc: 2.0 });
    setResult(null);
    setError(null);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setFactIdx((idx) => (idx + 1) % kidneyFacts.length);
    }, 8000);
    return () => clearInterval(interval);
  }, []);

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
            <div className="flex items-center space-x-3">
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
            </div>
            <motion.button 
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-4 py-2 text-sm font-medium text-blue-700 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors"
            >
              About / Model Info
            </motion.button>
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
                    <p className="text-sm text-slate-600 mt-1">Enter your test results below for prediction</p>
                  </div>
                </div>
              </motion.div>
              
              <div className="p-6">
                <form onSubmit={handleSubmit} className="space-y-8">
                  <div className="grid sm:grid-cols-2 gap-6">
                    {Object.entries(form).map(([key, value], index) => (
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
                            ({parameterInfo[key].range})
                          </span>
                        </label>
                        <input
                          type="number"
                          id={key}
                          name={key}
                          value={value}
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
                        result === 1 
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
                          {result === 1 ? "⚠️" : "✅"}
                        </motion.div>
                        <h3 className={`text-xl font-bold mb-2 ${
                          result === 1 ? "text-red-700" : "text-green-700"
                        }`}>
                          {result === 1 ? "High Risk: Kidney Stone Likely" : "Low Risk: No Kidney Stone Detected"}
                        </h3>
                        <p className="text-sm text-slate-700">
                          {result === 1 
                            ? "It's recommended to consult with a healthcare professional for further evaluation." 
                            : "Your parameters are within normal ranges. Keep maintaining healthy habits!"
                          }
                        </p>
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
                        Recent Predictions
                      </h3>
                      <button 
                        className="text-sm text-blue-600 hover:text-blue-700 font-medium" 
                        onClick={() => setHistory([])}
                      >
                        Clear History
                      </button>
                    </div>
                    <div className="overflow-hidden rounded-lg border border-slate-200">
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="bg-slate-100 text-slate-800">
                              <th className="px-4 py-3 text-left font-semibold">Time</th>
                              <th className="px-4 py-3 text-left font-semibold">Result</th>
                              <th className="px-4 py-3 text-left font-semibold">Parameters</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200">
                            {history.map((h, i) => (
                              <motion.tr 
                                key={i}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.05 }}
                                className="bg-white hover:bg-slate-50 transition-colors"
                              >
                                <td className="px-4 py-3 text-slate-600">{h.time}</td>
                                <td className="px-4 py-3">
                                  <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
                                    h.result === 1 
                                      ? "bg-red-100 text-red-700" 
                                      : "bg-green-100 text-green-700"
                                  }`}>
                                    {h.result === 1 ? "🚨 High Risk" : "✅ Low Risk"}
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-slate-600">
                                  <div className="flex flex-wrap gap-2">
                                    {Object.entries(h.form).map(([k, v]) => (
                                      <span key={k} className="inline-flex items-center text-xs bg-slate-100 px-2 py-1 rounded">
                                        <span className="font-medium">{parameterInfo[k].label}:</span>
                                        <span className="ml-1">{v}</span>
                                      </span>
                                    ))}
                                  </div>
                                </td>
                              </motion.tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
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
                  {Object.entries(parameterInfo).map(([key, info], index) => (
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
                          Normal: <span className="font-medium text-blue-600">{info.range}</span>
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