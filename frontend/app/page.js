// pages/index.jsx
"use client";
import { useState } from "react";

export default function Home() {
  const [formData, setFormData] = useState({
    diameter: "",
    height: "",
    species: "",
    condition: "",
    soil: "",
    siteFactor: "",
    weather: "",
    rootFailure: "",
    stemFailure: "",
    branchFailure: "",
    decayLocation: "",
    decayPresent: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Submitted:", formData);
    // TODO: send formData to your API/model endpoint
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center p-8">
      <div className="w-full max-w-3xl bg-gray-900 rounded-2xl shadow-xl p-8 space-y-8">
        <h1 className="text-3xl font-bold text-green-400 text-center">
          Tree Failure Analysis
        </h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Tree Basics */}
          <Section title="Tree Basics">
            <InputField
              label="Diameter of Tree (cm)"
              name="diameter"
              type="number"
              value={formData.diameter}
              onChange={handleChange}
            />
            <InputField
              label="Height of Tree (m)"
              name="height"
              type="number"
              value={formData.height}
              onChange={handleChange}
            />
            <SelectField
              label="Tree Species"
              name="species"
              value={formData.species}
              onChange={handleChange}
              options={[
                "Cedrus Atlantica",
                "Pinus Resinosa",
                "Robinia Pseudoacacia",
                "Quercus Robur",
                "Pseudotsuga Menziesii",
                "Acer Rubrum",
                "Ulmus Americana",
                "Prunus Avium",
                "Fraxinus Pennsylvanica",
                "Populus Trichocarpa",
                // ...add more from your dataset
              ]}
            />
            <SelectField
              label="Condition"
              name="condition"
              value={formData.condition}
              onChange={handleChange}
              options={["Good", "Fair", "Poor", "Dead"]}
            />
          </Section>

          {/* Environment */}
          <Section title="Environment">
            <SelectField
              label="Type of Soil"
              name="soil"
              value={formData.soil}
              onChange={handleChange}
              options={["Clay", "Loam", "Silt", "Dirt", "Sandy"]}
            />
            <SelectField
              label="Site Factors"
              name="siteFactor"
              value={formData.siteFactor}
              onChange={handleChange}
              options={[
                "None",
                "Urban Environment",
                "Lawn",
                "Natural Area",
                "Roots Restricted",
                "Steep Slope",
                "Soil Compaction",
              ]}
            />
            <SelectField
              label="Weather Factors"
              name="weather"
              value={formData.weather}
              onChange={handleChange}
              options={[
                "None",
                "Wind",
                "Rain",
                "Ice",
                "Snow",
                "High Temps",
                "Low Temps",
              ]}
            />
          </Section>

          {/* Failures */}
          <Section title="Failure Indicators">
            <SelectField
              label="Root Failure"
              name="rootFailure"
              value={formData.rootFailure}
              onChange={handleChange}
              options={[
                "None",
                "Mechanical Damage",
                "Broken Roots",
                "Cut Roots",
                "Decay Organism Present",
                "Soil Failure",
              ]}
            />
            <SelectField
              label="Stem Failure"
              name="stemFailure"
              value={formData.stemFailure}
              onChange={handleChange}
              options={[
                "None",
                "Topping",
                "Crack",
                "Cavity",
                "Decay Present",
                "Bulge",
              ]}
            />
            <SelectField
              label="Branch Failure"
              name="branchFailure"
              value={formData.branchFailure}
              onChange={handleChange}
              options={[
                "None",
                "Crack",
                "Decay Present",
                "Bulge",
                "Over-Extended Branch",
                "Break at Attachment",
              ]}
            />
          </Section>

          {/* Decay */}
          <Section title="Decay">
            <SelectField
              label="Location & % of Decay"
              name="decayLocation"
              value={formData.decayLocation}
              onChange={handleChange}
              options={[
                "None",
                "<25%",
                ">25%",
                ">50%",
                ">75%",
                "100%",
                "Heartwood",
                "Sapwood",
                "Canker",
                "Root",
              ]}
            />
            <SelectField
              label="Decay Present"
              name="decayPresent"
              value={formData.decayPresent}
              onChange={handleChange}
              options={[
                "None",
                "Phaeolus Schweinitzii",
                "Armillaria spp.",
                "Ganoderma Applanatum",
                "Phytophthora Cinnamomii",
                "Other",
              ]}
            />
          </Section>

          <button
            type="submit"
            className="w-full py-3 px-6 rounded-xl font-semibold bg-green-500 hover:bg-green-600 text-white shadow-lg"
          >
            Analyze Tree
          </button>
        </form>
      </div>
    </div>
  );
}

/* ---------------- Components ---------------- */

function Section({ title, children }) {
  const [open, setOpen] = useState(true);

  return (
    <div className="border border-gray-700 rounded-xl">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex justify-between items-center px-4 py-3 text-left font-semibold text-green-400"
      >
        {title}
        <span>{open ? "−" : "+"}</span>
      </button>
      {open && <div className="p-4 space-y-4">{children}</div>}
    </div>
  );
}

function InputField({ label, name, type, value, onChange }) {
  return (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
      />
    </div>
  );
}

function SelectField({ label, name, value, onChange, options }) {
  return (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <select
        name={name}
        value={value}
        onChange={onChange}
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
      >
        <option value="">Select...</option>
        {options.map((opt, i) => (
          <option key={i} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}
