import { useState, useRef, useCallback, useEffect, ReactNode } from "react";
import { initializeApp } from "firebase/app";
import {
  getFirestore, collection, addDoc,
  getDocs, orderBy, query, serverTimestamp,
  doc, updateDoc, deleteDoc,
} from "firebase/firestore";
import {
  PhotoIcon,
  SparklesIcon,
  PlusCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  XMarkIcon,
  MagnifyingGlassIcon,
  ArrowUpTrayIcon,
  CpuChipIcon,
  BoltIcon,
  TagIcon,
  PencilSquareIcon,
  BookmarkIcon,
  ArrowLeftIcon,
  TrashIcon,
  CalendarIcon,
  ChartBarIcon,
  InformationCircleIcon,
  MapPinIcon,
  ChatBubbleBottomCenterTextIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CheckCircleIcon,
  StarIcon,
  EyeIcon,
} from "@heroicons/react/24/outline";
import {
  CheckCircleIcon as CheckCircleSolid,
  SparklesIcon as SparklesSolid,
  XMarkIcon as XMarkSolid,
} from "@heroicons/react/24/solid";
import type { ComponentType, SVGProps } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────
type HeroIcon = ComponentType<SVGProps<SVGSVGElement> & { title?: string }>;

interface Pred       { flower: string; confidence: number }
interface Detection  { flower: string; confidence: number; icon: string; bbox: { x: number; y: number; w: number; h: number }; top5: Pred[] }
interface APIResult  { annotated_image: string; detections: Detection[]; top_flower: string; top_confidence: number; top_icon: string; model_used?: string; custom_profiles?: string[] }
interface HistoryItem {
  id: string;
  imageUrl: string;
  annotatedUrl: string;
  topFlower: string;
  confidence: number;
  timestamp: Date;
  notes?: string;
  location?: string;
  tags?: string[];
  correctedName?: string;
}
interface TrainStatus { status: string; message?: string; corrections: number; classes: string[]; sample_counts?: Record<string, number> }
interface CatData     { categories: string[]; custom_labels: string[]; custom_count: number }
interface SavedDetails { notes: string; location: string; tags: string[] }

type Step = "idle" | "uploading" | "classifying" | "done";
type Tab  = "classify" | "add" | "history";

// ── Firebase ──────────────────────────────────────────────────────────────────
const firebaseApp = initializeApp({
  apiKey: "AIzaSyAcebu-NMVHoVU-PvdznA4TxV67KIkjvcA",
  authDomain: "plantclassification-502b7.firebaseapp.com",
  projectId: "plantclassification-502b7",
  storageBucket: "plantclassification-502b7.firebasestorage.app",
  messagingSenderId: "515186234560",
  appId: "1:515186234560:web:04780d8e521b614678c925",
  measurementId: "G-GEZ1H1NPBQ",
});
const db = getFirestore(firebaseApp);

// ── Constants ─────────────────────────────────────────────────────────────────
const CLOUD_NAME    = "di3lbgjne";
const UPLOAD_PRESET = "plantclass";
const API_URL       = "https://floraai-8f8u.onrender.com";

// ── API helpers ───────────────────────────────────────────────────────────────
async function uploadToCloudinary(file: File): Promise<string> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("upload_preset", UPLOAD_PRESET);
  const r = await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`, { method: "POST", body: fd });
  const d = await r.json();
  if (!d.secure_url) throw new Error("Cloudinary upload failed");
  return d.secure_url as string;
}

async function uploadBase64ToCloudinary(dataUrl: string): Promise<string> {
  const fd = new FormData();
  fd.append("file", dataUrl);
  fd.append("upload_preset", UPLOAD_PRESET);
  const r = await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`, { method: "POST", body: fd });
  return ((await r.json()).secure_url as string) || "";
}

async function classifyViaBackend(file: File): Promise<APIResult> {
  const fd = new FormData();
  fd.append("image", file);
  const r = await fetch(`${API_URL}/classify`, { method: "POST", body: fd });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error((e as { error?: string }).error || `Server error ${r.status}`); }
  return r.json() as Promise<APIResult>;
}

async function triggerRetrain(): Promise<TrainStatus> {
  return (await fetch(`${API_URL}/retrain`, { method: "POST" })).json() as Promise<TrainStatus>;
}

async function fetchCategories(): Promise<CatData> {
  try {
    return await (await fetch(`${API_URL}/flower-categories`)).json() as CatData;
  } catch {
    try {
      const snap = await getDocs(collection(db, "flower_training_data"));
      const labels = [...new Set(snap.docs.map(d => d.data().label as string).filter(Boolean))].sort();
      return { categories: labels, custom_labels: labels, custom_count: labels.length };
    } catch {
      return { categories: [], custom_labels: [], custom_count: 0 };
    }
  }
}

async function saveToFirestore(data: Omit<HistoryItem, "id" | "timestamp">): Promise<string> {
  const ref = await addDoc(collection(db, "flower_classifications"), { ...data, timestamp: serverTimestamp() });
  return ref.id;
}

async function saveTrainingSample(label: string, imageUrl: string): Promise<void> {
  await addDoc(collection(db, "flower_training_data"), { label, imageUrl, timestamp: serverTimestamp() });
}

async function loadHistory(): Promise<HistoryItem[]> {
  const q = query(collection(db, "flower_classifications"), orderBy("timestamp", "desc"));
  const snap = await getDocs(q);
  return snap.docs.map(d => ({
    id: d.id,
    ...(d.data() as Omit<HistoryItem, "id" | "timestamp">),
    timestamp: (d.data().timestamp as { toDate?: () => Date })?.toDate?.() ?? new Date(),
  }));
}

async function updateFlowerRecord(id: string, updates: Partial<Omit<HistoryItem, "id" | "timestamp">>): Promise<void> {
  await updateDoc(doc(db, "flower_classifications", id), updates as Record<string, unknown>);
}

async function deleteFlowerRecord(id: string): Promise<void> {
  await deleteDoc(doc(db, "flower_classifications", id));
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function confColor(c: number): string {
  if (c >= 70) return "text-emerald-400";
  if (c >= 40) return "text-amber-400";
  return "text-red-400";
}
function confBarColor(c: number): string {
  if (c >= 70) return "bg-emerald-500";
  if (c >= 40) return "bg-amber-500";
  return "bg-red-500";
}
function capitalize(s: string): string {
  return s.split(" ").map((w: string) => w[0]?.toUpperCase() + w.slice(1)).join(" ");
}
function confLabel(c: number): string {
  if (c >= 85) return "Very High";
  if (c >= 70) return "High";
  if (c >= 50) return "Moderate";
  if (c >= 30) return "Low";
  return "Very Low";
}

// ── UI Primitives ─────────────────────────────────────────────────────────────
function Spinner({ className = "" }: { className?: string }) {
  return <ArrowPathIcon className={`animate-spin ${className}`} />;
}

function ConfidenceBar({ value, colorClass }: { value: number; colorClass: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-700">
      <div
        className={`h-full rounded-full transition-all duration-700 ${colorClass}`}
        style={{ width: `${Math.min(value, 100)}%` }}
      />
    </div>
  );
}

function FieldLabel({ icon: Icon, children }: { icon?: HeroIcon; children: ReactNode }) {
  return (
    <label className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-500">
      {Icon && <Icon className="h-3.5 w-3.5 shrink-0" />}
      {children}
    </label>
  );
}

function SectionCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-zinc-800 bg-zinc-900/60 ${className}`}>
      {children}
    </div>
  );
}

function StatusBadge({ confidence }: { confidence: number }) {
  const color =
    confidence >= 70 ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/25" :
    confidence >= 40 ? "bg-amber-500/15 text-amber-400 border-amber-500/25" :
                       "bg-red-500/15 text-red-400 border-red-500/25";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold backdrop-blur-sm ${color}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {confidence.toFixed(1)}% · {confLabel(confidence)}
    </span>
  );
}

// ── Inline Details Panel ──────────────────────────────────────────────────────
function InlineDetailsPanel({
  docId,
  onSaved,
}: {
  docId: string;
  onSaved: (details: SavedDetails) => void;
}) {
  const [open, setOpen]           = useState(false);
  const [notes, setNotes]         = useState("");
  const [location, setLocation]   = useState("");
  const [tagInput, setTagInput]   = useState("");
  const [tags, setTags]           = useState<string[]>([]);
  const [saving, setSaving]       = useState(false);
  const [savedData, setSavedData] = useState<SavedDetails | null>(null);
  const [editing, setEditing]     = useState(false);

  const addTag = () => {
    const t = tagInput.trim().toLowerCase();
    if (t && !tags.includes(t)) setTags((prev: string[]) => [...prev, t]);
    setTagInput("");
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const details: SavedDetails = { notes: notes.trim(), location: location.trim(), tags };
      await updateFlowerRecord(docId, details);
      setSavedData(details);
      setOpen(false);
      setEditing(false);
      onSaved(details);
    } catch (e) { console.error(e); }
    finally { setSaving(false); }
  };

  const handleEdit = () => {
    if (savedData) { setNotes(savedData.notes); setLocation(savedData.location); setTags(savedData.tags); }
    setEditing(true);
    setOpen(true);
  };

  if (savedData && !editing) {
    const hasAny = savedData.notes || savedData.location || savedData.tags.length > 0;
    if (!hasAny) return null;
    return (
      <SectionCard className="overflow-hidden">
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-2">
            <CheckCircleSolid className="h-4 w-4 text-emerald-400" />
            <span className="text-sm font-semibold text-zinc-200">Classification Details</span>
          </div>
          <button onClick={handleEdit}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-400 transition-colors hover:border-amber-500/40 hover:text-amber-400">
            <PencilSquareIcon className="h-3.5 w-3.5" />Edit
          </button>
        </div>
        <div className="space-y-3 p-4">
          {savedData.location && (
            <div className="flex items-start gap-2.5">
              <MapPinIcon className="mt-0.5 h-4 w-4 shrink-0 text-sky-400" />
              <div>
                <p className="mb-0.5 text-xs font-semibold uppercase tracking-wider text-zinc-600">Location</p>
                <p className="text-sm text-zinc-200">{savedData.location}</p>
              </div>
            </div>
          )}
          {savedData.notes && (
            <div className="flex items-start gap-2.5">
              <ChatBubbleBottomCenterTextIcon className="mt-0.5 h-4 w-4 shrink-0 text-sky-400" />
              <div>
                <p className="mb-0.5 text-xs font-semibold uppercase tracking-wider text-zinc-600">Notes</p>
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-200">{savedData.notes}</p>
              </div>
            </div>
          )}
          {savedData.tags.length > 0 && (
            <div className="flex items-start gap-2.5">
              <TagIcon className="mt-0.5 h-4 w-4 shrink-0 text-sky-400" />
              <div>
                <p className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-600">Tags</p>
                <div className="flex flex-wrap gap-1.5">
                  {savedData.tags.map((tag: string) => (
                    <span key={tag} className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 text-xs font-medium text-sky-400">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard className="overflow-hidden">
      <button onClick={() => setOpen((o: boolean) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-zinc-800/50">
        <div className="flex items-center gap-2">
          <ChatBubbleBottomCenterTextIcon className="h-4 w-4 shrink-0 text-sky-400" />
          <span className="text-sm font-medium text-zinc-400">
            {editing ? "Edit classification details" : "Add notes & details"}
          </span>
          {!editing && (
            <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-600">optional</span>
          )}
        </div>
        {open ? <ChevronUpIcon className="h-4 w-4 text-zinc-600" /> : <ChevronDownIcon className="h-4 w-4 text-zinc-600" />}
      </button>

      {open && (
        <div className="border-t border-zinc-800 p-4 space-y-4">
          <div>
            <FieldLabel icon={MapPinIcon}>Location</FieldLabel>
            <input value={location} onChange={(e) => setLocation(e.target.value)}
              placeholder="e.g. Batangas, UP campus, home garden…"
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20"
            />
          </div>
          <div>
            <FieldLabel icon={ChatBubbleBottomCenterTextIcon}>Observations / Notes</FieldLabel>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={3}
              placeholder="Color, smell, condition, context, environment…"
              className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20"
            />
          </div>
          <div>
            <FieldLabel icon={TagIcon}>Tags</FieldLabel>
            <div className="flex gap-2">
              <input value={tagInput} onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addTag(); } }}
                placeholder="Type a tag, press Enter to add…"
                className="flex-1 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-sky-500/50"
              />
              <button onClick={addTag}
                className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs font-semibold text-sky-400 transition-colors hover:bg-sky-500/20">
                Add Tag
              </button>
            </div>
            {tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {tags.map((tag: string) => (
                  <span key={tag} className="inline-flex items-center gap-1 rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 text-xs font-medium text-sky-400">
                    {tag}
                    <button onClick={() => setTags((prev: string[]) => prev.filter((t: string) => t !== tag))}
                      className="text-sky-600 hover:text-red-400 transition-colors">
                      <XMarkSolid className="h-3 w-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="flex gap-2 pt-1">
            <button onClick={handleSave}
              disabled={saving || (!notes.trim() && !location.trim() && tags.length === 0)}
              className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-sky-500 py-2.5 text-sm font-semibold text-zinc-950 transition-all hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-40">
              {saving ? <><Spinner className="h-4 w-4" />Saving…</> : <><BookmarkIcon className="h-4 w-4" />Save Details</>}
            </button>
            {editing && (
              <button onClick={() => { setEditing(false); setOpen(false); }}
                className="shrink-0 rounded-lg border border-zinc-700 px-4 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-200">
                Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </SectionCard>
  );
}

// ── Flower Detail Drawer ──────────────────────────────────────────────────────
function FlowerDetailDrawer({
  item,
  onClose,
  onSave,
  onDelete,
}: {
  item: HistoryItem;
  onClose: () => void;
  onSave: (updated: HistoryItem) => void;
  onDelete: (id: string) => void;
}) {
  const [editing, setEditing]             = useState(false);
  const [saving, setSaving]               = useState(false);
  const [deleting, setDeleting]           = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editName, setEditName]           = useState(item.correctedName ?? item.topFlower);
  const [editNotes, setEditNotes]         = useState(item.notes ?? "");
  const [editLocation, setEditLocation]   = useState(item.location ?? "");
  const [editTagInput, setEditTagInput]   = useState("");
  const [editTags, setEditTags]           = useState<string[]>(item.tags ?? []);
  const [showAnnotated, setShowAnnotated] = useState(true);

  const displayImg = showAnnotated && item.annotatedUrl ? item.annotatedUrl : item.imageUrl;

  const handleSave = async () => {
    setSaving(true);
    try {
      const updates: Partial<Omit<HistoryItem, "id" | "timestamp">> = {
        correctedName: editName.trim() || item.topFlower,
        notes: editNotes.trim(),
        location: editLocation.trim(),
        tags: editTags,
      };
      await updateFlowerRecord(item.id, updates);
      onSave({ ...item, ...updates });
      setEditing(false);
    } catch (e) { console.error(e); }
    finally { setSaving(false); }
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await deleteFlowerRecord(item.id);
      onDelete(item.id);
      onClose();
    } catch (e) { console.error(e); setDeleting(false); }
  };

  const addTag = () => {
    const t = editTagInput.trim().toLowerCase();
    if (t && !editTags.includes(t)) setEditTags((prev: string[]) => [...prev, t]);
    setEditTagInput("");
  };

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="fixed inset-x-0 bottom-0 z-50 sm:inset-y-0 sm:left-auto sm:right-0 sm:w-[520px] flex flex-col overflow-hidden rounded-t-2xl sm:rounded-none border-t sm:border-t-0 sm:border-l border-zinc-800 bg-zinc-950 shadow-2xl max-h-[90vh] sm:max-h-none">

        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-5 py-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-emerald-500/25 bg-emerald-500/10">
              <InformationCircleIcon className="h-5 w-5 text-emerald-400" />
            </div>
            <div className="min-w-0">
              <h2 className="truncate text-base font-bold text-white">Flower Record Details</h2>
              <p className="text-xs text-zinc-600">View &amp; edit classification data</p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {!editing && (
              <button onClick={() => setEditing(true)}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-400 transition-all hover:border-amber-500/40 hover:bg-amber-500/8 hover:text-amber-400">
                <PencilSquareIcon className="h-3.5 w-3.5" />Edit Record
              </button>
            )}
            <button onClick={onClose} aria-label="Close panel"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-800 text-zinc-500 transition-all hover:border-red-500/30 hover:text-red-400">
              <XMarkIcon className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto">
          {/* Image */}
          <div className="relative h-56 w-full overflow-hidden bg-zinc-900">
            <img src={displayImg} alt={item.correctedName ?? item.topFlower} className="h-full w-full object-cover" />
            <div className="absolute left-3 top-3">
              <StatusBadge confidence={item.confidence} />
            </div>
            {item.annotatedUrl && item.annotatedUrl !== item.imageUrl && (
              <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 overflow-hidden rounded-full border border-white/15 bg-black/80 p-0.5 backdrop-blur-sm">
                {(["Annotated", "Original"] as const).map((label, idx) => (
                  <button key={label} onClick={() => setShowAnnotated(idx === 0)}
                    className={`rounded-full px-3 py-1.5 text-xs font-semibold transition-all ${(idx === 0 ? showAnnotated : !showAnnotated) ? "bg-emerald-500 text-zinc-950" : "text-zinc-400 hover:text-zinc-200"}`}>
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-5 p-5">
            {/* Name */}
            <div>
              <FieldLabel icon={SparklesIcon}>
                {editing ? "Flower Name (Editable)" : "Identified Flower"}
              </FieldLabel>
              {editing ? (
                <>
                  <input value={editName} onChange={(e) => setEditName(e.target.value)}
                    placeholder="Enter correct flower name…"
                    className="w-full rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-3 text-lg font-bold text-zinc-100 outline-none transition-colors focus:border-emerald-500/50"
                  />
                  <p className="mt-1.5 text-xs text-zinc-600">
                    Original: <span className="capitalize text-zinc-500">{item.topFlower}</span>
                  </p>
                </>
              ) : (
                <>
                  <h3 className="text-2xl font-bold capitalize text-white">{item.correctedName ?? item.topFlower}</h3>
                  {item.correctedName && item.correctedName !== item.topFlower && (
                    <p className="mt-1 text-xs text-zinc-600">
                      Originally classified as: <span className="capitalize text-zinc-500">{item.topFlower}</span>
                    </p>
                  )}
                </>
              )}
            </div>

            {/* Metadata grid */}
            <div className="grid grid-cols-2 gap-3">
              <SectionCard className="p-3.5">
                <div className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-600">
                  <CalendarIcon className="h-3.5 w-3.5 shrink-0" />Date Classified
                </div>
                <p className="text-sm font-semibold text-zinc-200">
                  {item.timestamp instanceof Date
                    ? item.timestamp.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
                    : "—"}
                </p>
                <p className="mt-0.5 text-xs text-zinc-600">
                  {item.timestamp instanceof Date ? item.timestamp.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }) : ""}
                </p>
              </SectionCard>
              <SectionCard className="p-3.5">
                <div className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-600">
                  <ChartBarIcon className="h-3.5 w-3.5 shrink-0" />Confidence Score
                </div>
                <p className={`text-sm font-bold ${confColor(item.confidence)}`}>{item.confidence.toFixed(2)}%</p>
                <div className="mt-2">
                  <ConfidenceBar value={item.confidence} colorClass={confBarColor(item.confidence)} />
                </div>
              </SectionCard>
            </div>

            <div className="h-px w-full bg-zinc-800" />

            {/* Location */}
            <div>
              <FieldLabel icon={MapPinIcon}>Location / Observation Site</FieldLabel>
              {editing ? (
                <input value={editLocation} onChange={(e) => setEditLocation(e.target.value)}
                  placeholder="e.g. Batangas, Quezon City, campus garden…"
                  className="w-full rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-emerald-500/50"
                />
              ) : (
                <p className={`text-sm ${item.location ? "text-zinc-200" : "italic text-zinc-700"}`}>
                  {item.location || "No location recorded"}
                </p>
              )}
            </div>

            {/* Notes */}
            <div>
              <FieldLabel icon={ChatBubbleBottomCenterTextIcon}>Field Notes / Observations</FieldLabel>
              {editing ? (
                <textarea value={editNotes} onChange={(e) => setEditNotes(e.target.value)} rows={4}
                  placeholder="Color, smell, condition, habitat, context…"
                  className="w-full resize-none rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-emerald-500/50"
                />
              ) : (
                <p className={`whitespace-pre-wrap text-sm leading-relaxed ${item.notes ? "text-zinc-200" : "italic text-zinc-700"}`}>
                  {item.notes || "No notes recorded"}
                </p>
              )}
            </div>

            {/* Tags */}
            <div>
              <FieldLabel icon={TagIcon}>Classification Tags</FieldLabel>
              {editing ? (
                <div>
                  <div className="flex gap-2">
                    <input value={editTagInput} onChange={(e) => setEditTagInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addTag(); } }}
                      placeholder="Type a tag, press Enter to add…"
                      className="flex-1 rounded-xl border border-zinc-700 bg-zinc-800 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-emerald-500/50"
                    />
                    <button onClick={addTag}
                      className="shrink-0 rounded-xl border border-emerald-500/25 bg-emerald-500/10 px-3 py-2 text-xs font-semibold text-emerald-400 hover:bg-emerald-500/20">
                      Add Tag
                    </button>
                  </div>
                  {editTags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {editTags.map((tag: string) => (
                        <span key={tag} className="inline-flex items-center gap-1 rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 text-xs font-medium text-sky-400">
                          {tag}
                          <button onClick={() => setEditTags((prev: string[]) => prev.filter((t: string) => t !== tag))}
                            className="text-sky-600 hover:text-red-400">
                            <XMarkSolid className="h-3 w-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {(item.tags ?? []).length > 0
                    ? item.tags!.map((tag: string) => (
                        <span key={tag} className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 text-xs font-medium text-sky-400">
                          {tag}
                        </span>
                      ))
                    : <span className="italic text-sm text-zinc-700">No tags added</span>}
                </div>
              )}
            </div>

            <div className="h-px w-full bg-zinc-800" />

            {/* Image links */}
            <div>
              <FieldLabel icon={PhotoIcon}>Stored Image URLs</FieldLabel>
              <div className="space-y-2">
                <a href={item.imageUrl} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-2 truncate rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-sky-400 transition-colors hover:border-sky-500/30 hover:text-sky-300">
                  <PhotoIcon className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">Original: {item.imageUrl}</span>
                </a>
                {item.annotatedUrl && item.annotatedUrl !== item.imageUrl && (
                  <a href={item.annotatedUrl} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 truncate rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-emerald-400 transition-colors hover:border-emerald-500/30 hover:text-emerald-300">
                    <SparklesIcon className="h-3.5 w-3.5 shrink-0" />
                    <span className="truncate">Annotated: {item.annotatedUrl}</span>
                  </a>
                )}
              </div>
            </div>

            {/* Save / Cancel while editing */}
            {editing && (
              <div className="flex gap-2">
                <button onClick={handleSave} disabled={saving}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-emerald-500 py-3 text-sm font-bold text-zinc-950 transition-all hover:bg-emerald-400 disabled:opacity-40">
                  {saving ? <><Spinner className="h-4 w-4" />Saving…</> : <><CheckCircleSolid className="h-4 w-4" />Save Changes</>}
                </button>
                <button onClick={() => { setEditing(false); setEditName(item.correctedName ?? item.topFlower); setEditNotes(item.notes ?? ""); setEditLocation(item.location ?? ""); setEditTags(item.tags ?? []); }}
                  className="shrink-0 rounded-xl border border-zinc-700 bg-zinc-800 px-5 py-3 text-sm font-medium text-zinc-400 transition-colors hover:text-zinc-200">
                  Cancel
                </button>
              </div>
            )}

            {/* Danger zone */}
            <div className="rounded-xl border border-red-500/15 bg-red-500/5 p-4">
              <p className="mb-2.5 text-xs font-bold uppercase tracking-wider text-red-500/60">Danger Zone</p>
              {!confirmDelete ? (
                <button onClick={() => setConfirmDelete(true)}
                  className="flex items-center gap-2 rounded-lg border border-red-500/20 px-3 py-2 text-xs font-medium text-red-500/70 transition-all hover:border-red-500/40 hover:bg-red-500/8 hover:text-red-400">
                  <TrashIcon className="h-3.5 w-3.5" />Delete this record permanently
                </button>
              ) : (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-red-400">Are you sure? This action cannot be undone.</p>
                  <div className="flex gap-2">
                    <button onClick={handleDelete} disabled={deleting}
                      className="flex items-center gap-1.5 rounded-lg bg-red-500 px-3 py-1.5 text-xs font-bold text-white transition-colors hover:bg-red-400 disabled:opacity-50">
                      <TrashIcon className="h-3.5 w-3.5" />
                      {deleting ? "Deleting…" : "Yes, Delete Record"}
                    </button>
                    <button onClick={() => setConfirmDelete(false)}
                      className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-300">
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function FlowerClassifier() {
  const [tab, setTab]               = useState<Tab>("classify");
  const [dragOver, setDragOver]     = useState(false);
  const [step, setStep]             = useState<Step>("idle");
  const [preview, setPreview]       = useState<string | null>(null);
  const [annotated, setAnnotated]   = useState<string | null>(null);
  const [result, setResult]         = useState<APIResult | null>(null);
  const [cloudUrl, setCloudUrl]     = useState<string | null>(null);
  const [error, setError]           = useState<string | null>(null);
  const [history, setHistory]       = useState<HistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [selectedFile, setSelectedFile]     = useState<File | null>(null);
  const [showAnnotated, setShowAnnotated]   = useState(true);
  const [currentDocId, setCurrentDocId]     = useState<string | null>(null);

  const [correcting, setCorrecting]             = useState(false);
  const [correctInput, setCorrectInput]         = useState("");
  const [suggestions, setSuggestions]           = useState<string[]>([]);
  const [corrected, setCorrected]               = useState<string | null>(null);
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [correctionSaved, setCorrectionSaved]   = useState(false);
  const [isNewFlower, setIsNewFlower]           = useState(false);
  const [trainStatus, setTrainStatus]           = useState<TrainStatus | null>(null);
  const [isRetraining, setIsRetraining]         = useState(false);

  const [allCategories, setAllCategories]         = useState<string[]>([]);
  const [customLabels, setCustomLabels]           = useState<string[]>([]);
  const [categoriesLoading, setCategoriesLoading] = useState(true);

  const [newFlowerName, setNewFlowerName]         = useState("");
  const [newFlowerFiles, setNewFlowerFiles]       = useState<File[]>([]);
  const [newFlowerPreviews, setNewFlowerPreviews] = useState<string[]>([]);
  const [addingFlower, setAddingFlower]           = useState(false);
  const [addFlowerDone, setAddFlowerDone]         = useState(false);
  const [addFlowerError, setAddFlowerError]       = useState<string | null>(null);

  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);

  const inputRef   = useRef<HTMLInputElement>(null);
  const correctRef = useRef<HTMLInputElement>(null);
  const addFileRef = useRef<HTMLInputElement>(null);

  const refreshCategories = useCallback(async () => {
    setCategoriesLoading(true);
    try {
      const data = await fetchCategories();
      setAllCategories(data.categories);
      setCustomLabels(data.custom_labels);
    } finally { setCategoriesLoading(false); }
  }, []);

  useEffect(() => { refreshCategories(); }, [refreshCategories]);
  useEffect(() => {
    loadHistory().then(setHistory).catch(console.error).finally(() => setLoadingHistory(false));
  }, []);

  useEffect(() => {
    if (!isRetraining) return;
    const id = setInterval(async () => {
      try {
        const d = await (await fetch(`${API_URL}/training-status`)).json() as { is_training: boolean };
        if (!d.is_training) { setIsRetraining(false); clearInterval(id); refreshCategories(); }
      } catch { /* keep polling */ }
    }, 3000);
    return () => clearInterval(id);
  }, [isRetraining, refreshCategories]);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) { setError("Please upload an image file (JPG, PNG, WEBP)."); return; }
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setAnnotated(null); setResult(null); setError(null);
    setStep("idle"); setShowAnnotated(true); setCloudUrl(null); setCurrentDocId(null);
    setCorrecting(false); setCorrectInput(""); setCorrected(null);
    setCorrectionSaved(false); setSuggestions([]); setTrainStatus(null); setIsNewFlower(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleClassify = async () => {
    if (!selectedFile) return;
    setError(null); setResult(null); setAnnotated(null); setCurrentDocId(null);
    setCorrecting(false); setCorrected(null); setCorrectionSaved(false); setTrainStatus(null);
    try {
      setStep("uploading");
      const url = await uploadToCloudinary(selectedFile);
      setCloudUrl(url);
      setStep("classifying");
      const api = await classifyViaBackend(selectedFile);
      let annUrl = url;
      try { annUrl = await uploadBase64ToCloudinary(api.annotated_image) || url; } catch { /* fallback */ }
      const docId = await saveToFirestore({ imageUrl: url, annotatedUrl: annUrl, topFlower: api.top_flower, confidence: api.top_confidence });
      setCurrentDocId(docId);
      loadHistory().then(setHistory).catch(console.error);
      setResult(api);
      setAnnotated(api.annotated_image);
      setStep("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setStep("idle");
    }
  };

  const handleCorrectInput = (val: string) => {
    setCorrectInput(val);
    if (val.length < 2) { setSuggestions([]); setIsNewFlower(false); return; }
    setSuggestions(allCategories.filter((c: string) => c.toLowerCase().includes(val.toLowerCase())).slice(0, 6));
    setIsNewFlower(!allCategories.some((c: string) => c.toLowerCase() === val.toLowerCase().trim()) && val.trim().length > 2);
  };

  const handleSaveCorrection = async () => {
    if (!correctInput.trim() || !cloudUrl) return;
    setSavingCorrection(true);
    try {
      await saveTrainingSample(correctInput.trim().toLowerCase(), cloudUrl);
      if (currentDocId) await updateFlowerRecord(currentDocId, { correctedName: correctInput.trim().toLowerCase() });
      setCorrected(correctInput.trim());
      setCorrectionSaved(true); setCorrecting(false); setSuggestions([]); setIsNewFlower(false);
      refreshCategories();
      setIsRetraining(true);
      const tr = await triggerRetrain();
      setTrainStatus(tr);
      if (tr.status !== "training_started") setIsRetraining(false);
    } catch (e: unknown) {
      setError("Could not save correction: " + (e instanceof Error ? e.message : "Unknown error"));
      setIsRetraining(false);
    } finally { setSavingCorrection(false); }
  };

  const reset = () => {
    setPreview(null); setAnnotated(null); setResult(null); setSelectedFile(null);
    setError(null); setStep("idle"); setCloudUrl(null); setCurrentDocId(null);
    setCorrecting(false); setCorrectInput(""); setCorrected(null);
    setCorrectionSaved(false); setSuggestions([]); setTrainStatus(null);
    setIsRetraining(false); setIsNewFlower(false);
  };

  const handleAddFiles = (files: FileList | null) => {
    if (!files) return;
    const arr = Array.from(files).filter((f: File) => f.type.startsWith("image/")).slice(0, 10);
    setNewFlowerFiles((prev: File[]) => [...prev, ...arr].slice(0, 10));
    arr.forEach((f: File) => {
      const r = new FileReader();
      r.onload = (e) => setNewFlowerPreviews((p: string[]) => [...p, e.target?.result as string].slice(0, 10));
      r.readAsDataURL(f);
    });
  };

  const removeAddFile = (i: number) => {
    setNewFlowerFiles((f: File[]) => f.filter((_: File, idx: number) => idx !== i));
    setNewFlowerPreviews((p: string[]) => p.filter((_: string, idx: number) => idx !== i));
  };

  const handleAddNewFlower = async () => {
    const name = newFlowerName.trim().toLowerCase();
    if (!name) { setAddFlowerError("Please enter the flower name."); return; }
    if (newFlowerFiles.length < 3) { setAddFlowerError(`Please add at least ${3 - newFlowerFiles.length} more photo(s).`); return; }
    setAddingFlower(true); setAddFlowerError(null);
    try {
      for (const file of newFlowerFiles) {
        const url = await uploadToCloudinary(file);
        await saveTrainingSample(name, url);
      }
      await refreshCategories();
      const tr = await triggerRetrain();
      setAddFlowerDone(true);
      setNewFlowerName(""); setNewFlowerFiles([]); setNewFlowerPreviews([]);
      if (tr.status === "training_started") setIsRetraining(true);
    } catch (e: unknown) {
      setAddFlowerError("Upload failed: " + (e instanceof Error ? e.message : "Unknown error"));
    } finally { setAddingFlower(false); }
  };

  const isProcessing = step === "uploading" || step === "classifying";
  const displayImg   = showAnnotated && annotated ? annotated : preview;

  const tabs: { id: Tab; label: string; shortLabel: string; icon: HeroIcon; activeClass: string }[] = [
    { id: "classify", label: "Classify Flower",  shortLabel: "Classify", icon: SparklesIcon,   activeClass: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" },
    { id: "add",      label: "Add New Variety",  shortLabel: "Add",      icon: PlusCircleIcon, activeClass: "bg-violet-500/15 text-violet-400 border-violet-500/30"   },
    { id: "history",  label: "History",          shortLabel: "History",  icon: ClockIcon,      activeClass: "bg-sky-500/15 text-sky-400 border-sky-500/30"             },
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Ambient BG */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-48 left-1/2 h-[500px] w-[500px] -translate-x-1/2 rounded-full bg-emerald-500/6 blur-3xl" />
        <div className="absolute top-1/2 -right-32 h-64 w-64 rounded-full bg-teal-500/4 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-8 sm:py-14">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header className="mb-8 sm:mb-10 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-4 py-1.5">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
            <span className="text-xs font-semibold uppercase tracking-widest text-emerald-400">
              cv2 · Oxford 102 · Dynamic ML
            </span>
          </div>
          <h1 className="mb-3 text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
            Flower{" "}
            <span className="bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">Vision</span>
          </h1>
          <p className="text-sm text-zinc-500">
            {categoriesLoading ? (
              <span className="inline-flex items-center gap-1.5">
                <Spinner className="h-3 w-3 text-zinc-600" />
                Loading flower categories…
              </span>
            ) : allCategories.length === 0 ? (
              <span className="inline-flex items-center justify-center gap-1.5 text-amber-500/80">
                <ExclamationTriangleIcon className="h-3.5 w-3.5" />
                No categories found — backend may be offline
              </span>
            ) : (
              <span className="inline-flex flex-wrap items-center justify-center gap-x-3 gap-y-1">
                <span className="inline-flex items-center gap-1 text-zinc-400">
                  <StarIcon className="h-3.5 w-3.5 text-emerald-400" />
                  <strong className="text-emerald-400">{allCategories.length}</strong> flower categories loaded
                </span>
                {customLabels.length > 0 && (
                  <span className="inline-flex items-center gap-1 text-violet-400">
                    <SparklesSolid className="h-3.5 w-3.5" />
                    <strong>{customLabels.length}</strong> custom varieties
                  </span>
                )}
              </span>
            )}
          </p>
        </header>

        {/* ── Tab Navigation ───────────────────────────────────────────────── */}
        <nav aria-label="Main navigation" className="mb-6">
          <div className="flex gap-1 rounded-2xl border border-zinc-800 bg-zinc-900 p-1">
            {tabs.map((t) => {
              const Icon = t.icon;
              const active = tab === t.id;
              return (
                <button key={t.id}
                  role="tab"
                  aria-selected={active}
                  onClick={() => { setTab(t.id); if (t.id === "add") setAddFlowerDone(false); }}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm font-semibold transition-all border
                    ${active ? t.activeClass : "text-zinc-500 hover:text-zinc-300 border-transparent hover:bg-zinc-800/50"}`}>
                  <Icon className="h-4 w-4 shrink-0" />
                  <span className="hidden sm:inline">{t.label}</span>
                  <span className="sm:hidden">{t.shortLabel}</span>
                  {t.id === "history" && history.length > 0 && (
                    <span className={`rounded-full px-1.5 py-0.5 text-xs font-bold leading-none ${active ? "bg-sky-500/20 text-sky-400" : "bg-zinc-800 text-zinc-500"}`}>
                      {history.length}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </nav>

        {/* ═══════════════════════════════════════════════════════════════════
            CLASSIFY TAB
        ═══════════════════════════════════════════════════════════════════ */}
        {tab === "classify" && (
          <main aria-label="Flower classification">
            <SectionCard className="overflow-hidden">
              {!preview ? (
                <div
                  role="button"
                  aria-label="Upload flower image — click or drag and drop"
                  tabIndex={0}
                  className={`m-4 sm:m-5 flex cursor-pointer flex-col items-center justify-center gap-5 rounded-xl border-2 border-dashed p-10 sm:p-16 text-center transition-all duration-300
                    ${dragOver ? "border-emerald-400/60 bg-emerald-500/8" : "border-zinc-700 hover:border-emerald-500/40 hover:bg-emerald-500/5"}`}
                  onDragOver={(e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => inputRef.current?.click()}
                  onKeyDown={(e: React.KeyboardEvent<HTMLDivElement>) => e.key === "Enter" && inputRef.current?.click()}
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-emerald-500/25 bg-emerald-500/10">
                    <PhotoIcon className="h-8 w-8 text-emerald-400" />
                  </div>
                  <div>
                    <p className="mb-1.5 text-base font-semibold text-zinc-200">Drop your flower photo here</p>
                    <p className="text-sm text-zinc-600">
                      or <span className="font-semibold text-emerald-400">click to browse files</span>
                    </p>
                    <p className="mt-2 text-xs text-zinc-700">Supports JPG · PNG · WEBP</p>
                  </div>
                  <input ref={inputRef} type="file" accept="image/*" className="sr-only"
                    aria-label="Select flower image file"
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                  />
                </div>
              ) : (
                <>
                  <div className="flex flex-col md:grid md:grid-cols-2">
                    {/* Image panel */}
                    <div className="relative overflow-hidden bg-zinc-900" style={{ height: "280px" }}>
                      <img src={displayImg ?? undefined} alt="Flower being classified" className="h-full w-full object-contain" />
                      {annotated && (
                        <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 overflow-hidden rounded-full border border-white/15 bg-black/80 p-0.5 backdrop-blur-sm">
                          {(["Detected View", "Original Photo"] as const).map((label, idx) => (
                            <button key={label} onClick={() => setShowAnnotated(idx === 0)}
                              aria-pressed={idx === 0 ? showAnnotated : !showAnnotated}
                              className={`rounded-full px-3 py-1.5 text-xs font-semibold transition-all ${(idx === 0 ? showAnnotated : !showAnnotated) ? "bg-emerald-500 text-zinc-950" : "text-zinc-400 hover:text-zinc-200"}`}>
                              {label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Results panel */}
                    <div className="flex flex-col gap-4 overflow-y-auto p-4 sm:p-5 md:max-h-[400px]">
                      {result ? (
                        <>
                          <div>
                            <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-zinc-600">
                              {corrected ? "Corrected Classification" : "Primary Match"}
                            </p>
                            <h2 className={`text-xl sm:text-2xl font-bold tracking-tight ${corrected ? "text-emerald-400" : "text-white"}`}>
                              {result.top_icon} {corrected ? capitalize(corrected) : result.top_flower}
                            </h2>
                            <p className="mt-1 text-sm text-zinc-500">
                              {corrected ? "Manually corrected · saved for retraining" : `${result.top_confidence.toFixed(1)}% confidence score`}
                            </p>
                          </div>

                          {result.model_used && (
                            <div className="flex items-center gap-1.5 text-xs text-zinc-600">
                              <CpuChipIcon className="h-3.5 w-3.5 shrink-0" />
                              Model: {result.model_used}
                            </div>
                          )}

                          {result.custom_profiles && result.custom_profiles.length > 0 && (
                            <div className="flex flex-wrap gap-1.5">
                              {result.custom_profiles.map((c: string) => (
                                <span key={c} className="inline-flex items-center gap-1 rounded-full border border-amber-500/25 bg-amber-500/10 px-2.5 py-0.5 text-xs font-semibold text-amber-400">
                                  <BoltIcon className="h-3 w-3" />{capitalize(c)}
                                </span>
                              ))}
                            </div>
                          )}

                          <div className="h-px w-full bg-zinc-800" />

                          {/* Top 5 */}
                          <div>
                            <p className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-600">
                              <ChartBarIcon className="h-3.5 w-3.5" />Top 5 Predictions
                            </p>
                            <div className="space-y-2.5">
                              {result.detections[0]?.top5.map((p: Pred, i: number) => (
                                <div key={i}>
                                  <div className="mb-1 flex items-center justify-between">
                                    <span className="text-xs font-medium capitalize text-zinc-300">
                                      {i === 0 && <span className="mr-1 text-emerald-500">●</span>}
                                      {p.flower}
                                    </span>
                                    <span className={`text-xs font-bold ${confColor(p.confidence)}`}>{p.confidence.toFixed(1)}%</span>
                                  </div>
                                  <ConfidenceBar value={p.confidence} colorClass={confBarColor(p.confidence)} />
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="h-px w-full bg-zinc-800" />

                          {currentDocId && (
                            <InlineDetailsPanel
                              docId={currentDocId}
                              onSaved={(details: SavedDetails) => {
                                setHistory((prev: HistoryItem[]) => prev.map((h: HistoryItem) =>
                                  h.id === currentDocId ? { ...h, ...details } : h
                                ));
                              }}
                            />
                          )}

                          {!correcting && !correctionSaved && (
                            <button onClick={() => { setCorrecting(true); setTimeout(() => correctRef.current?.focus(), 60); }}
                              className="flex w-full items-center justify-center gap-2 rounded-xl border border-zinc-700 bg-zinc-800/50 py-2.5 text-xs font-semibold text-zinc-500 transition-all hover:border-amber-500/30 hover:bg-amber-500/8 hover:text-amber-400">
                              <PencilSquareIcon className="h-3.5 w-3.5" />
                              Wrong result? Submit correction for AI retraining
                            </button>
                          )}

                          {correcting && (
                            <div className="rounded-xl border border-amber-500/25 bg-amber-500/8 p-4">
                              <p className="mb-3 flex items-center gap-2 text-xs font-bold text-amber-400">
                                <PencilSquareIcon className="h-3.5 w-3.5" />
                                Submit Correct Flower Name
                              </p>
                              <FieldLabel>Correct Flower Name</FieldLabel>
                              <div className="relative mb-3">
                                <input ref={correctRef}
                                  placeholder="Type the correct flower name…"
                                  value={correctInput}
                                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleCorrectInput(e.target.value)}
                                  onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && handleSaveCorrection()}
                                  className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-amber-500/50"
                                />
                                {suggestions.length > 0 && (
                                  <div className="absolute left-0 right-0 top-full z-50 mt-1 overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl">
                                    {suggestions.map((s: string) => (
                                      <button key={s} onClick={() => { setCorrectInput(s); setSuggestions([]); setIsNewFlower(false); }}
                                        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-sm capitalize text-zinc-400 transition-colors hover:bg-amber-500/10 hover:text-amber-400">
                                        <SparklesSolid className="h-3.5 w-3.5 shrink-0 text-amber-500/50" />{s}
                                      </button>
                                    ))}
                                  </div>
                                )}
                              </div>

                              {isNewFlower && correctInput.trim().length > 2 && (
                                <div className="mb-3 flex items-start gap-2 rounded-lg border border-violet-500/25 bg-violet-500/8 p-3 text-xs text-violet-400">
                                  <SparklesSolid className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                                  New variety! "<strong className="text-violet-300">{capitalize(correctInput.trim())}</strong>" will be added to the classifier.
                                </div>
                              )}

                              <div className="flex gap-2">
                                <button onClick={handleSaveCorrection} disabled={savingCorrection || !correctInput.trim()}
                                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-amber-500 py-2.5 text-xs font-bold text-zinc-950 transition-all hover:bg-amber-400 disabled:opacity-40">
                                  {savingCorrection ? <><Spinner className="h-3.5 w-3.5" />Saving…</>
                                    : isNewFlower ? <><SparklesSolid className="h-3.5 w-3.5" />Add New: "{capitalize(correctInput.trim()) || "…"}"</>
                                    : <><BookmarkIcon className="h-3.5 w-3.5" />Save as Training Sample</>}
                                </button>
                                <button onClick={() => { setCorrecting(false); setSuggestions([]); setIsNewFlower(false); }}
                                  className="shrink-0 rounded-lg border border-zinc-700 px-3 py-2 text-xs font-medium text-zinc-500 hover:text-zinc-300">
                                  Cancel
                                </button>
                              </div>
                            </div>
                          )}

                          {correctionSaved && (
                            <div className="flex items-center gap-2 rounded-xl border border-emerald-500/25 bg-emerald-500/8 p-3 text-sm font-semibold text-emerald-400">
                              <CheckCircleSolid className="h-4 w-4 shrink-0" />
                              Saved! "{capitalize(corrected ?? "")}" added to training data.
                            </div>
                          )}

                          {isRetraining && (
                            <div className="flex items-center gap-2 rounded-xl border border-violet-500/25 bg-violet-500/8 p-3 text-sm font-semibold text-violet-400">
                              <Spinner className="h-4 w-4 shrink-0" />
                              Retraining model in background…
                            </div>
                          )}

                          {!isRetraining && trainStatus?.status === "not_enough_data" && (
                            <div className="flex items-center gap-2 rounded-xl border border-amber-500/25 bg-amber-500/8 p-3 text-sm text-amber-400">
                              <ExclamationTriangleIcon className="h-4 w-4 shrink-0" />
                              {trainStatus.message}
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="flex flex-1 flex-col items-center justify-center gap-4 py-10 text-center">
                          {isProcessing ? (
                            <>
                              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-emerald-500/25 bg-emerald-500/10">
                                <Spinner className="h-7 w-7 text-emerald-400" />
                              </div>
                              <div>
                                <p className="text-sm font-semibold text-zinc-300" aria-live="polite">
                                  {step === "uploading" ? "Uploading image…" : "Classifying flower…"}
                                </p>
                                <p className="mt-1 text-xs text-zinc-600">
                                  {step === "uploading" ? "Sending to Cloudinary" : "Running computer vision model"}
                                </p>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-zinc-700 bg-zinc-800">
                                <MagnifyingGlassIcon className="h-7 w-7 text-zinc-600" />
                              </div>
                              <div>
                                <p className="text-sm font-semibold text-zinc-400">Ready to classify</p>
                                <p className="mt-1 text-xs text-zinc-700">
                                  Press <span className="font-bold text-emerald-400">Detect &amp; Classify</span> below
                                </p>
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {isProcessing && (
                    <div className="flex items-center gap-2.5 border-t border-zinc-800 bg-zinc-900/50 px-5 py-3 text-xs text-zinc-500" aria-live="polite">
                      <Spinner className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
                      {step === "uploading" ? "Uploading image to Cloudinary…" : "Running cv2 detection pipeline…"}
                    </div>
                  )}

                  {step === "done" && result && (
                    <div className="flex items-center gap-2 border-t border-zinc-800 bg-emerald-500/5 px-5 py-3 text-xs text-emerald-400">
                      <CheckCircleSolid className="h-3.5 w-3.5 shrink-0" />
                      Classification complete · {result.detections.length} region{result.detections.length !== 1 ? "s" : ""} detected · Saved to Firebase
                    </div>
                  )}

                  {error && (
                    <div className="mx-4 mb-4 flex items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/8 p-4 text-sm text-red-400" role="alert">
                      <ExclamationTriangleIcon className="mt-0.5 h-4 w-4 shrink-0" />
                      <div>
                        <p className="font-semibold">Classification Error</p>
                        <p className="mt-1 text-xs text-red-400/70">
                          {error.includes("fetch") || error.includes("Failed")
                            ? "Cannot connect to Flask backend. Make sure it's running on localhost:5000."
                            : error}
                        </p>
                        {(error.includes("fetch") || error.includes("Failed")) && (
                          <code className="mt-1.5 block rounded bg-red-500/10 px-2 py-1 font-mono text-xs text-red-500/80">
                            cd backend && python3 app.py
                          </code>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-3 border-t border-zinc-800 px-4 sm:px-5 py-4">
                    <button onClick={handleClassify} disabled={isProcessing || result !== null}
                      aria-label={isProcessing ? "Processing…" : result ? "Already classified" : "Detect and classify flower"}
                      className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-emerald-500 py-3 text-sm font-bold text-zinc-950 transition-all hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40">
                      {isProcessing
                        ? <><Spinner className="h-4 w-4" />{step === "uploading" ? "Uploading…" : "Classifying…"}</>
                        : result
                        ? <><CheckCircleSolid className="h-4 w-4" />Classification Complete</>
                        : <><SparklesIcon className="h-4 w-4" />Detect &amp; Classify</>}
                    </button>
                    <button onClick={reset} aria-label="Upload a new photo"
                      className="flex items-center gap-2 rounded-xl border border-zinc-700 bg-zinc-800 px-4 sm:px-5 text-sm font-semibold text-zinc-400 transition-all hover:border-zinc-600 hover:text-zinc-200">
                      <ArrowLeftIcon className="h-4 w-4 shrink-0" />
                      <span className="hidden sm:inline">New Photo</span>
                    </button>
                  </div>
                </>
              )}
            </SectionCard>
          </main>
        )}

        {/* ═══════════════════════════════════════════════════════════════════
            ADD FLOWER TAB
        ═══════════════════════════════════════════════════════════════════ */}
        {tab === "add" && (
          <main aria-label="Add new flower variety">
            <SectionCard className="overflow-hidden">
              <div className="flex items-center gap-4 border-b border-zinc-800 px-5 py-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-violet-500/25 bg-violet-500/10">
                  <PlusCircleIcon className="h-5 w-5 text-violet-400" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white">Add New Flower Variety</h2>
                  <p className="text-xs text-zinc-600">Upload photos to teach the classifier a new flower type</p>
                </div>
              </div>

              <div className="p-5 sm:p-6">
                <div className="mb-6 grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {([
                    { n: "1", label: "Enter flower name", icon: PencilSquareIcon },
                    { n: "2", label: "Upload 3–10 photos", icon: PhotoIcon },
                    { n: "3", label: "Click Add Variety",  icon: PlusCircleIcon },
                    { n: "4", label: "Model auto-retrains", icon: SparklesIcon },
                  ] as { n: string; label: string; icon: HeroIcon }[]).map((s) => {
                    const Icon = s.icon;
                    return (
                      <div key={s.n} className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2">
                        <span className="shrink-0 text-sm font-black text-violet-400">{s.n}</span>
                        <Icon className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                        <span className="text-xs text-zinc-500 leading-tight">{s.label}</span>
                      </div>
                    );
                  })}
                </div>

                {addFlowerDone ? (
                  <div className="flex flex-col items-center gap-5 rounded-xl border border-emerald-500/25 bg-emerald-500/8 p-8 text-center">
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-emerald-500/25 bg-emerald-500/12">
                      <CheckCircleSolid className="h-8 w-8 text-emerald-400" />
                    </div>
                    <div>
                      <h3 className="mb-1 text-lg font-bold text-emerald-400">Variety Added Successfully!</h3>
                      <p className="text-sm text-zinc-500">The classifier has been retrained with the new flower data.</p>
                    </div>
                    <button onClick={() => { setAddFlowerDone(false); setTab("classify"); }}
                      className="flex items-center gap-2 rounded-xl bg-emerald-500 px-6 py-3 text-sm font-bold text-zinc-950 transition-all hover:bg-emerald-400">
                      <SparklesIcon className="h-4 w-4" />Try Classifying Now
                    </button>
                  </div>
                ) : (
                  <>
                    {addFlowerError && (
                      <div className="mb-5 flex items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/8 p-4 text-sm text-red-400" role="alert">
                        <ExclamationTriangleIcon className="mt-0.5 h-4 w-4 shrink-0" />
                        {addFlowerError}
                      </div>
                    )}

                    <div className="mb-5">
                      <FieldLabel icon={TagIcon}>Flower Name / Variety</FieldLabel>
                      <input
                        placeholder="e.g. Sampaguita, Waling-Waling, Jade Vine…"
                        value={newFlowerName}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewFlowerName(e.target.value)}
                        aria-label="Flower name or variety"
                        className="w-full rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition-colors focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20"
                      />
                    </div>

                    <div className="mb-4">
                      <FieldLabel icon={PhotoIcon}>Training Photos (minimum 3, maximum 10)</FieldLabel>
                      <div
                        role="button"
                        aria-label="Upload training photos"
                        tabIndex={0}
                        onClick={() => addFileRef.current?.click()}
                        onKeyDown={(e: React.KeyboardEvent<HTMLDivElement>) => e.key === "Enter" && addFileRef.current?.click()}
                        className="flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed border-zinc-700 bg-zinc-900/50 p-8 text-center transition-all hover:border-violet-500/40 hover:bg-violet-500/5">
                        <ArrowUpTrayIcon className="h-8 w-8 text-violet-400/60" />
                        <div>
                          <p className="text-sm font-medium text-zinc-400">
                            Drag photos here, or <span className="font-semibold text-violet-400">browse files</span>
                          </p>
                          <p className="mt-1 text-xs text-zinc-600">JPG · PNG · WEBP · up to 10 photos</p>
                        </div>
                        <input ref={addFileRef} type="file" accept="image/*" multiple className="sr-only"
                          aria-label="Select training photos"
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleAddFiles(e.target.files)}
                        />
                      </div>
                    </div>

                    {newFlowerPreviews.length > 0 && (
                      <div className="mb-5">
                        <div className="flex flex-wrap gap-2 mb-2">
                          {newFlowerPreviews.map((src: string, i: number) => (
                            <div key={i} className="relative overflow-hidden rounded-lg border border-zinc-700" style={{ height: "72px", width: "72px" }}>
                              <img src={src} alt={`Training photo ${i + 1}`} className="h-full w-full object-cover" />
                              <button onClick={() => removeAddFile(i)} aria-label={`Remove photo ${i + 1}`}
                                className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-black/80 text-red-400 transition-colors hover:bg-black hover:text-red-300">
                                <XMarkIcon className="h-3 w-3" />
                              </button>
                            </div>
                          ))}
                        </div>
                        <p className="flex items-center gap-2 text-xs text-zinc-600">
                          <PhotoIcon className="h-3.5 w-3.5 shrink-0" />
                          <strong className="text-violet-400">{newFlowerFiles.length}</strong> photo{newFlowerFiles.length !== 1 ? "s" : ""} selected
                          {newFlowerFiles.length < 3
                            ? <span className="text-red-400">— {3 - newFlowerFiles.length} more required</span>
                            : <span className="inline-flex items-center gap-1 text-emerald-400"><CheckCircleIcon className="h-3.5 w-3.5" />Minimum met</span>}
                        </p>
                      </div>
                    )}

                    <button onClick={handleAddNewFlower}
                      disabled={addingFlower || !newFlowerName.trim() || newFlowerFiles.length < 3}
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 py-3.5 text-sm font-bold text-white transition-all hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40">
                      {addingFlower
                        ? <><Spinner className="h-4 w-4" />Uploading photos &amp; saving…</>
                        : <><SparklesSolid className="h-4 w-4" />Add "{newFlowerName.trim() || "…"}" to Classifier</>}
                    </button>
                  </>
                )}

                {customLabels.length > 0 && (
                  <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                    <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-600">
                      <TagIcon className="h-3.5 w-3.5" />
                      Custom Varieties in System ({customLabels.length})
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {customLabels.map((l: string) => (
                        <span key={l} className="rounded-full border border-violet-500/25 bg-violet-500/10 px-2.5 py-0.5 text-xs font-medium capitalize text-violet-400">
                          {l}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </SectionCard>
          </main>
        )}

        {/* ═══════════════════════════════════════════════════════════════════
            HISTORY TAB
        ═══════════════════════════════════════════════════════════════════ */}
        {tab === "history" && (
          <main aria-label="Classification history">
            <SectionCard className="overflow-hidden">
              <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
                <div className="flex items-center gap-2.5">
                  <ClockIcon className="h-5 w-5 text-sky-400" />
                  <h2 className="text-base font-bold text-white">Classification History</h2>
                </div>
                <span className="rounded-full border border-sky-500/25 bg-sky-500/10 px-2.5 py-1 text-xs font-semibold text-sky-400">
                  {history.length} record{history.length !== 1 ? "s" : ""}
                </span>
              </div>

              {loadingHistory ? (
                <div className="flex items-center gap-3 px-6 py-10 text-sm text-zinc-500" role="status" aria-live="polite">
                  <Spinner className="h-4 w-4 shrink-0 text-sky-400" />
                  Loading classification history…
                </div>
              ) : history.length === 0 ? (
                <div className="flex flex-col items-center gap-4 py-16 text-center">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-zinc-700 bg-zinc-800">
                    <ClockIcon className="h-7 w-7 text-zinc-600" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-zinc-500">No classifications yet</p>
                    <p className="mt-1 text-xs text-zinc-700">Classify a flower to see it here</p>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 sm:gap-4 p-4 sm:p-5">
                  {history.map((item: HistoryItem) => (
                    <button key={item.id} onClick={() => setSelectedItem(item)}
                      aria-label={`View details for ${item.correctedName ?? item.topFlower}`}
                      className="group overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/70 text-left transition-all duration-200 hover:-translate-y-1 hover:border-sky-500/30 hover:shadow-xl hover:shadow-sky-500/8">
                      <div className="relative overflow-hidden bg-zinc-900" style={{ aspectRatio: "4/3" }}>
                        <img src={item.annotatedUrl || item.imageUrl}
                          alt={`Classified: ${item.correctedName ?? item.topFlower}`}
                          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                          <div className="flex items-center gap-1.5 rounded-full border border-white/20 bg-black/60 px-3 py-1.5 backdrop-blur-sm">
                            <EyeIcon className="h-3.5 w-3.5 text-white" />
                            <span className="text-xs font-semibold text-white">View Details</span>
                          </div>
                        </div>
                        {(item.tags ?? []).length > 0 && (
                          <div className="absolute left-2 top-2">
                            <span className="flex items-center gap-0.5 rounded-full border border-sky-500/30 bg-black/60 px-1.5 py-0.5 text-xs font-medium text-sky-300 backdrop-blur-sm">
                              <TagIcon className="h-2.5 w-2.5" />
                              {item.tags!.length}
                            </span>
                          </div>
                        )}
                      </div>
                      <div className="p-3">
                        <p className="mb-0.5 truncate text-sm font-bold capitalize text-zinc-100">
                          {item.correctedName ?? item.topFlower}
                        </p>
                        {item.correctedName && item.correctedName !== item.topFlower && (
                          <p className="mb-1 truncate text-xs capitalize text-zinc-600 line-through">{item.topFlower}</p>
                        )}
                        {item.location && (
                          <p className="mb-1.5 flex items-center gap-1 truncate text-xs text-zinc-600">
                            <MapPinIcon className="h-3 w-3 shrink-0" />{item.location}
                          </p>
                        )}
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-zinc-700">
                            {item.timestamp instanceof Date
                              ? item.timestamp.toLocaleDateString("en-US", { month: "short", day: "numeric" })
                              : ""}
                          </span>
                          <span className={`rounded-full px-2 py-0.5 text-xs font-bold bg-zinc-800 ${confColor(item.confidence)}`}>
                            {item.confidence.toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </SectionCard>
          </main>
        )}

        <p className="mt-8 text-center text-xs text-zinc-800">
          cv2 · HOG+HSV · Oxford 102 + Dynamic Custom Categories · Cloudinary · Firebase
        </p>
      </div>

      {selectedItem && (
        <FlowerDetailDrawer
          item={selectedItem}
          onClose={() => setSelectedItem(null)}
          onSave={(updated: HistoryItem) => {
            setHistory((prev: HistoryItem[]) => prev.map((h: HistoryItem) => h.id === updated.id ? updated : h));
            setSelectedItem(updated);
          }}
          onDelete={(id: string) => {
            setHistory((prev: HistoryItem[]) => prev.filter((h: HistoryItem) => h.id !== id));
            setSelectedItem(null);
          }}
        />
      )}
    </div>
  );
}
