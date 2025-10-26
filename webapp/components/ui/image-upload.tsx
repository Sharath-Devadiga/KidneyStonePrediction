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
      <div
        className={`relative h-64 rounded-lg border-2 border-dashed transition-colors cursor-pointer ${
          dragActive ? "border-blue-500 bg-blue-50" : "border-slate-300"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={openFileSystem} // <-- Make the whole area clickable
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          ref={fileInputRef} // <-- Assign ref
          className="hidden" // <-- Hide the input
          disabled={isLoading}
        />
        
        {preview ? (
          <div className="absolute inset-0 flex items-center justify-center p-4 pointer-events-none">
            <Image
              src={preview}
              alt="Preview"
              layout="fill"
              objectFit="contain"
              className="rounded"
            />
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center pointer-events-none">
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="text-4xl mb-3"
            >
              📷
            </motion.div>
            <p className="text-sm font-medium text-slate-700">
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <motion.span
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="inline-block w-4 h-4 border-2 border-slate-300 border-t-slate-600 rounded-full"
                  />
                  Processing image...
                </span>
              ) : (
                "Drag and drop your image here"
              )}
            </p>
            <p className="mt-1.5 text-xs text-slate-500">
              Supported formats: JPEG, PNG
            </p>
            {/* Add an explicit button */}
            {!isLoading && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation(); // <-- Prevent double click
                  openFileSystem();
                }}
                className="mt-4 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors pointer-events-auto"
              >
                Click to browse
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}