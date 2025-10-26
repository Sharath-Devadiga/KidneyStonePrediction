"use client";

import { useState, useRef } from "react"; // <-- Import useRef
import { motion } from "framer-motion";
import Image from "next/image";

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  isLoading?: boolean;
}

export function ImageUpload({ onImageUpload, isLoading = false }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null); // <-- Add ref for file input

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        return;
      }
      handleFile(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target?.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        return;
      }
      handleFile(file);
    }
  };

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    onImageUpload(file);
  };

  // Function to trigger file input click
  const openFileSystem = () => {
    if (isLoading) return;
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`relative h-72 rounded-xl border-2 border-dashed transition-all cursor-pointer overflow-hidden ${
          dragActive 
            ? "border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 scale-105 shadow-xl" 
            : "border-slate-300 bg-gradient-to-br from-white to-slate-50 hover:border-blue-400 hover:shadow-lg"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={openFileSystem}
        whileHover={{ scale: preview ? 1 : 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          ref={fileInputRef}
          className="hidden"
          disabled={isLoading}
        />
        
        {preview ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute inset-0 flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 to-blue-50"
          >
            <div className="relative w-full h-full flex items-center justify-center">
              <Image
                src={preview}
                alt="Preview"
                layout="fill"
                objectFit="contain"
                className="rounded-lg"
              />
              {!isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm rounded-full p-2 shadow-lg"
                >
                  <span className="text-green-600 text-xl">✓</span>
                </motion.div>
              )}
            </div>
          </motion.div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="relative"
            >
              <motion.div
                animate={dragActive ? { 
                  scale: [1, 1.2, 1],
                  rotate: [0, 5, -5, 0]
                } : {}}
                transition={{ duration: 0.5 }}
                className="text-6xl mb-4"
              >
                📷
              </motion.div>
              {dragActive && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -inset-4 bg-blue-500/20 rounded-full blur-xl"
                />
              )}
            </motion.div>
            
            <p className="text-base font-semibold text-slate-800 mb-2">
              {isLoading ? (
                <span className="flex items-center gap-3">
                  <motion.span
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="inline-block w-5 h-5 border-3 border-slate-300 border-t-blue-600 rounded-full"
                  />
                  Processing image...
                </span>
              ) : dragActive ? (
                <span className="text-blue-600">Drop your image here!</span>
              ) : (
                "Drag and drop your image here"
              )}
            </p>
            
            {!isLoading && !dragActive && (
              <>
                <p className="text-sm text-slate-500 mb-4">
                  or click to browse from your device
                </p>
                
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="flex items-center gap-2 text-xs text-slate-400 bg-slate-100 px-4 py-2 rounded-full"
                >
                  <span>📁</span>
                  <span>Supported: JPEG, PNG</span>
                </motion.div>
              </>
            )}
          </div>
        )}
      </motion.div>
    </div>
  );
}