

import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-blue-100 to-blue-200">
      {/* Navigation */}
      <nav className="bg-white/90 backdrop-blur-sm shadow-sm py-4">
        <div className="max-w-7xl mx-auto px-4 flex justify-between items-center">
          <Link href="/" className="text-xl font-bold text-blue-700 hover:text-blue-800 transition-colors">
            Kidney Stone Predictor
          </Link>
          <div className="flex items-center gap-4">
            <Link href="/predict">
              <Button variant="ghost">Predict</Button>
            </Link>
            <Link href="/about">
              <Button variant="ghost">About</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 py-16 md:py-24">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <h1 className="text-4xl md:text-6xl font-extrabold">
              <span className="bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
                Predict Kidney Stone Risk
              </span>
            </h1>
            <p className="text-xl text-gray-600 leading-relaxed">
              Get instant, AI-powered predictions based on your urine analysis. Our advanced machine learning model helps identify potential kidney stone risks quickly and accurately.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Link href="/predict">
                <Button className="w-full sm:w-auto text-lg px-8 py-6 bg-blue-600 hover:bg-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl">
                  Start Prediction
                </Button>
              </Link>
              <Link href="/about">
                <Button variant="outline" className="w-full sm:w-auto text-lg px-8 py-6 border-2 hover:bg-blue-50 transition-all duration-200">
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
          
          {/* Feature Cards */}
          <div className="grid sm:grid-cols-2 gap-6">
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-200">
              <div className="text-3xl mb-4">⚡</div>
              <h3 className="text-xl font-bold text-blue-800 mb-2">Fast Analysis</h3>
              <p className="text-gray-600">Get instant predictions using our advanced machine learning model.</p>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-200">
              <div className="text-3xl mb-4">📊</div>
              <h3 className="text-xl font-bold text-blue-800 mb-2">Track History</h3>
              <p className="text-gray-600">Keep track of all your predictions and monitor changes over time.</p>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-200">
              <div className="text-3xl mb-4">🔒</div>
              <h3 className="text-xl font-bold text-blue-800 mb-2">Reliable Results</h3>
              <p className="text-gray-600">Built on proven medical parameters and advanced algorithms.</p>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-200">
              <div className="text-3xl mb-4">📱</div>
              <h3 className="text-xl font-bold text-blue-800 mb-2">Modern UI</h3>
              <p className="text-gray-600">Professional interface that works on all your devices.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white/90 backdrop-blur-sm border-t mt-auto py-8">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-gray-600">&copy; {new Date().getFullYear()} Kidney Stone Prediction App</p>
          <p className="text-sm text-gray-500 mt-2">Built with FastAPI, Next.js, and modern ML technology</p>
        </div>
      </footer>
    </main>
  );
}