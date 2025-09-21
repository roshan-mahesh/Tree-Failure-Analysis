// pages/index.jsx
"use client";
import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [formData, setFormData] = useState({
    treeSpecies: "",
    diameter: "",
    height: "",
    condition: "",
    siteFactors: [],
    soilType: "",
    weatherFactors: [],
    rootFailure: [],
    stemFailure: [],
    branchFailure: [],
    locationOfDecay: "",
    decayAmount: "",
    decayType: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((p) => ({ ...p, [name]: value }));
  };

  const toggleMulti = (field, option) => {
    setFormData((p) => {
      const current = p[field] || [];
      const exists = current.includes(option);
      const next = exists ? current.filter((o) => o !== option) : [...current, option];
      return { ...p, [field]: next };
    });
  };

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const payload = {
      treeSpecies: formData.treeSpecies,
      diameter: Number(formData.diameter),
      height: Number(formData.height),
      condition: formData.condition,
      siteFactors: formData.siteFactors,
      soilType: formData.soilType,
      weatherFactors: formData.weatherFactors,
      rootFailure: formData.rootFailure,
      stemFailure: formData.stemFailure,
      branchFailure: formData.branchFailure,
      locationOfDecay: formData.locationOfDecay,
      decayAmount: formData.decayAmount,
      decayType: formData.decayType,
    };

    try {
      const response = await fetch("https://tree-failure-analysis.onrender.com/api/evaluate_tree", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error during submission:", error);
      setResult({ error: "An error occurred. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  // --- Options ---
  const speciesOptions = [
    "Cedrus Atlantica",
    "Pinus Resinosa",
    "Robinia Pseudoacacia",
    "Cupressus Nootkatensis",
    "Prunus Salicina",
    "Tilia Cordata",
    "Picea Omorika",
    "Pinus Monticola",
    "Pseudotsuga Menziesii",
    "Acer Rubrum",
    "Ulumus Americana",
    "Prunus Avium",
    "Crataegus Laevigata",
    "Quercus Robur",
    "Cercidiphyllum Japonicum",
    "Fraxinus Pennsylvanica",
    "Fraxinus Latifolia",
    "Quercus Rubra",
    "Quercus Ilex",
    "Populus Nigra",
    "Chamecyparis Pisifera",
    "Acer Platanoides",
    "Alnus Rubra",
    "Fraxinus Oxycarpa",
    "Liriodendron Tulipifera",
    "Pyrus Calleryana",
    "Salix Babylonica",
    "Abies Grandis",
    "Pinus Sylvestris",
    "Liquidambar Styraciflua",
    "Acer Palmatum",
    "Cercis Occidentalis",
    "Quercus Coccinea",
    "Pinus Thunbergii",
    "Calocedrus Decurrens",
    "Acer Circinatum",
    "Pinus Contorta",
    "Acer Macrophyllum",
    "Malus Pumilla",
    "Acer Saccharinum",
    "Taxus Baccata",
    "Pinus Nigra",
    "Corylus Colurna",
    "Salix Scouleriana",
    "Catalpa Bignonioides",
    "Populus Deltoids",
    "Salix Hookeriana",
    "Prunus Cerasifera",
    "Quercus Palustris",
    "Picea Glauca",
    "Prunus Laurocerasus",
    "Taxus Brevifolia",
    "Prunus Serrulata",
    "Quercus Alba",
    "Arbutus Menziesii",
    "Chamaecyparis Lawsoniana",
    "Cedrus Deodara",
    "Picea Pungens",
    "Prunus X",
    "Ulmus Procera",
    "Salix Lasiandra",
    "Betula Papyrifera",
    "Quercus Garryana",
    "Prunus Emarginata",
    "Picea Abies",
    "Populus Tremuloides",
    "Tilia Tomentosa",
    "Cornus Kuosa",
    "Thuja Plicata",
    "Cornus Nuttallii",
    "Populus Trichocarpa",
    "Populus Alba",
    "Cupressus X",
    "Tsuga Heterophylla",
    "Fraxinus Americana",
  ];

  const conditionOptions = ["Good", "Fair", "Poor", "Dead"];
  const siteFactorOptions = ["None", "Urban Environment", "Removal of Nearby Tree", "Soil Compaction", "Grade Changes", "Roots Restricted", "Steep Slope", "Soil Eroded", "Lawn", "Natural Area"];
  const soilTypeOptions = ["Dirt", "Clay", "Loam", "Silt", "Sandy"];
  const weatherOptions = ["None", "Wind", "Rain", "Ice", "Snow", "High Temps", "Low Temps"];
  const rootFailureOptions = ["Mechanical Damage", "Broken Roots", "Gridled Roots", "Cut Roots", "Surface Roots Wounded", "Root Plate Lifted", "Soil Failure", "Other"];
  const stemFailureOptions = ["Topping", "Seam", "Bulge", "Crack", "Cavity", "Decay Present", "Dead Stem", "Included Bark", "Other"];
  const branchFailureOptions = ["Lion Tailing", "Seam", "Old Pruning Wound at Failure Point", "Codominant Attachment", "Cavity", "Branch was Dead", "Mechanical Damage", "Over Extended Branch", "Break at Attachment", "Other"];
  const locationOfDecayOptions = ["None", "Root", "Sapwood", "Heartwood", "Canker", "Other"];
  const decayAmountOptions = ["None", "<25%", "25-50%", "50-75%", ">75%", "100%"];
  const decayTypeOptions = ["None", "Phaeolus Schweinitzii", "Kretzchmaria Duesta", "Phellinus Weirii", "Perenniporia Subacida", "Heterobasidion Occidentale", "Ceriporopsis Rivulosa", "Porodadalea Pini", "Ganoderma Applanatum", "Neofusicoccum Arbuti", "Ganoderma Brownii", "Phytophthora Cinnamomii", "Phytophthora Cactorum", "Armillaria spp.", "Phellinus Hartigii", "Other"];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl bg-gray-900 rounded-2xl shadow-xl p-8 space-y-6">
        <header className="mb-2">
          <h1 className="text-3xl font-bold text-green-400">Tree Failure Analysis</h1>
          <p className="text-sm text-gray-400 mt-1">Fill the form below and submit to run the model.</p>
        </header>

        <form onSubmit={handleSubmit} className="space-y-6">
          <Section title="Tree Basics">
            <SelectField label="Tree Species" name="treeSpecies" value={formData.treeSpecies} onChange={handleChange} options={speciesOptions} />
            <NumberField label="Diameter of Tree (cm)" name="diameter" value={formData.diameter} onChange={handleChange} />
            <NumberField label="Height of Tree (m)" name="height" value={formData.height} onChange={handleChange} />
            <SelectField label="Condition" name="condition" value={formData.condition} onChange={handleChange} options={conditionOptions} />
          </Section>

          <Section title="Environment">
            <MultiSelectField label="Site Factors" options={siteFactorOptions} selected={formData.siteFactors} onToggle={(opt) => toggleMulti("siteFactors", opt)} />
            <SelectField label="Type of Soil" name="soilType" value={formData.soilType} onChange={handleChange} options={soilTypeOptions} />
            <MultiSelectField label="Weather Factors" options={weatherOptions} selected={formData.weatherFactors} onToggle={(opt) => toggleMulti("weatherFactors", opt)} />
          </Section>

          <Section title="Failure Indicators">
            <MultiSelectField label="Root Failure" options={rootFailureOptions} selected={formData.rootFailure} onToggle={(opt) => toggleMulti("rootFailure", opt)} />
            <MultiSelectField label="Stem Failure" options={stemFailureOptions} selected={formData.stemFailure} onToggle={(opt) => toggleMulti("stemFailure", opt)} />
            <MultiSelectField label="Branch Failure" options={branchFailureOptions} selected={formData.branchFailure} onToggle={(opt) => toggleMulti("branchFailure", opt)} />
          </Section>

          <Section title="Decay">
            <SelectField label="Location of Decay" name="locationOfDecay" value={formData.locationOfDecay} onChange={handleChange} options={locationOfDecayOptions} />
            <SelectField label="Decay Amount" name="decayAmount" value={formData.decayAmount} onChange={handleChange} options={decayAmountOptions} />
            <SelectField label="Type of Decay" name="decayType" value={formData.decayType} onChange={handleChange} options={decayTypeOptions} />
          </Section>

          <div className="flex space-x-3">
            <button
              type="submit"
              disabled={loading}
              className={`flex-1 py-3 rounded-xl font-semibold text-white ${
                loading ? "bg-gray-600 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
              }`}
            >
              {loading ? "Processing..." : "Analyze Tree"}
            </button>
            <button
              type="button"
              onClick={() => {
                setFormData({
                  treeSpecies: "",
                  diameter: "",
                  height: "",
                  condition: "",
                  siteFactors: [],
                  soilType: "",
                  weatherFactors: [],
                  rootFailure: [],
                  stemFailure: [],
                  branchFailure: [],
                  locationOfDecay: "",
                  decayAmount: "",
                  decayType: "",
                });
              }}
              className="py-3 px-5 rounded-xl font-medium bg-gray-800 text-gray-200 border border-gray-700"
            >
              Reset
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-6 p-4 rounded-xl border border-gray-700 bg-gray-800 space-y-2">
            {result.error ? (
              <p className="text-red-400 font-semibold">{result.error}</p>
            ) : (
              <>
                <h2 className="text-xl font-bold text-green-400">Prediction Results</h2>
                <p>
                  <strong>Root Failure Probability:</strong>{" "}
                  <span className="text-green-300">{(result.rootFailureProbability * 100).toFixed(2)}%</span>
                </p>
                <p>
                  <strong>Stem Failure Probability:</strong>{" "}
                  <span className="text-green-300">{(result.stemFailureProbability * 100).toFixed(2)}%</span>
                </p>
                <p>
                  <strong>Branch Failure Probability:</strong>{" "}
                  <span className="text-green-300">{(result.branchFailureProbability * 100).toFixed(2)}%</span>
                </p>
              </>
            )}
          </div>
        )}

        {/* Overlay while processing */}
        {loading && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-gray-900 p-6 rounded-xl shadow-xl text-center space-y-3">
              <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-green-400 mx-auto"></div>
              <p className="text-green-300 font-semibold">Processing your request...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- Components ---------------- */
function Section({ title, children }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-gray-700 rounded-xl">
      <button type="button" onClick={() => setOpen((s) => !s)} className="w-full flex justify-between items-center px-4 py-3 text-left font-semibold text-green-300">
        {title}
        <span className="text-xl">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="p-4 space-y-4">{children}</div>}
    </div>
  );
}

function NumberField({ label, name, value, onChange }) {
  return (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <input
        type="number"
        name={name}
        value={value}
        onChange={onChange}
        onWheel={(e) => e.target.blur()} // disable scroll increment/decrement
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
      />
    </div>
  );
}

function SelectField({ label, name, value, onChange, options = [] }) {
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

function MultiSelectField({ label, options = [], selected = [], onToggle }) {
  const [open, setOpen] = useState(false);
  const ref = useRef();

  useEffect(() => {
    function onDocClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("click", onDocClick);
    return () => document.removeEventListener("click", onDocClick);
  }, []);

  return (
    <div className="space-y-1 relative" ref={ref}>
      <label className="text-sm font-medium">{label}</label>
      <button type="button" onClick={() => setOpen((s) => !s)} className="w-full text-left px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg flex items-center justify-between">
        <div className="flex-1 min-w-0">
          {selected.length === 0 ? <span className="text-gray-400">Select...</span> : (
            <div className="flex flex-wrap gap-2">
              {selected.map((s, idx) => (
                <span key={idx} className="text-xs px-2 py-1 bg-green-900/60 rounded-full">{s}</span>
              ))}
            </div>
          )}
        </div>
        <svg className="w-5 h-5 text-gray-300" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.293l3.71-4.06a.75.75 0 011.14.98l-4.25 4.65a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z" clipRule="evenodd" />
        </svg>
      </button>

      {open && (
        <div className="absolute left-0 right-0 mt-2 bg-gray-800 border border-gray-700 rounded-lg max-h-56 overflow-auto z-20 p-3">
          {options.map((opt, i) => {
            const checked = selected.includes(opt);
            return (
              <label key={i} className="flex items-center space-x-3 py-1">
                <input type="checkbox" checked={checked} onChange={() => onToggle(opt)} className="form-checkbox h-4 w-4 text-green-500 bg-gray-900 rounded" />
                <span className="text-sm">{opt}</span>
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}
