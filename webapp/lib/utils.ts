import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// API Types
export interface UrineParameters {
  gravity: number
  ph: number
  osmo: number
  cond: number
  urea: number
  calc: number
}

export interface PredictionResult {
  prediction: number
}

export interface ModelMetadata {
  model_version: string
  model_type: string
  input_parameters: {
    [key: string]: {
      min: number
      max: number
      unit: string
    }
  }
  output: {
    type: string
    values: {
      [key: string]: string
    }
  }
}

// Parameter information with ranges and labels
export const parameterInfo = {
  gravity: {
    label: "Specific Gravity",
    range: "1.005 - 1.035 sg",
    defaultValue: 1.020
  },
  ph: {
    label: "pH Level",
    range: "4.5 - 8.0 pH",
    defaultValue: 6.0
  },
  osmo: {
    label: "Osmolarity",
    range: "150 - 1200 mOsm/kg",
    defaultValue: 700
  },
  cond: {
    label: "Conductivity",
    range: "5 - 40 mS/cm",
    defaultValue: 21.0
  },
  urea: {
    label: "Urea Concentration",
    range: "100 - 600 mmol/L",
    defaultValue: 350
  },
  calc: {
    label: "Calcium Level",
    range: "0.5 - 15 mmol/L",
    defaultValue: 4.0
  }
}

// Error handling function
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return String(error)
}

// Validation function for urine parameters
export function validateParameters(params: Partial<UrineParameters>): string | null {
  for (const [key, value] of Object.entries(params)) {
    const info = parameterInfo[key as keyof UrineParameters]
    if (!info) continue

    const [min, max] = info.range.split(" - ").map(s => 
      parseFloat(s.split(" ")[0])
    )

    if (value < min || value > max) {
      return `${info.label} must be between ${min} and ${max}`
    }
  }
  return null
}
