import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "file:///C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "D:/codexproject/可视话作业";
const OUT = `${ROOT}/slides`;
const TMP = `${ROOT}/outputs/ppt_preview`;
const FINAL = `${OUT}/VAST2023_MC1_非法捕鱼知识图谱可视分析课程展示.pptx`;
const data = JSON.parse(await fs.readFile(`${ROOT}/outputs/ppt_data.json`, "utf-8"));

await fs.mkdir(OUT, { recursive: true });
await fs.mkdir(TMP, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  bg: "#F8FAFC",
  ink: "#0F172A",
  muted: "#64748B",
  blue: "#2563EB",
  blue2: "#DBEAFE",
  red: "#DC2626",
  red2: "#FEE2E2",
  amber: "#F59E0B",
  green: "#059669",
  line: "#CBD5E1",
  white: "#FFFFFF",
};
const frame = { left: 64, top: 46, width: 1152, height: 620 };

function addBox(slide, { left, top, width, height, fill = C.white, line = C.line, radius = "rounded-lg" }) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addText(slide, text, pos, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: style.typeface ?? "Microsoft YaHei",
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? C.ink,
    ...style,
  };
  return shape;
}

function title(slide, t, sub = "") {
  addText(slide, t, { left: frame.left, top: 34, width: 820, height: 48 }, { fontSize: 32, bold: true });
  if (sub) addText(slide, sub, { left: frame.left, top: 82, width: 980, height: 30 }, { fontSize: 15, color: C.muted });
}

function footer(slide, idx) {
  addText(slide, `VAST Challenge 2023 MC1  |  ${idx}`, { left: 1040, top: 684, width: 170, height: 20 }, { fontSize: 11, color: C.muted });
}

function bulletList(slide, items, left, top, width, fontSize = 21, gap = 46) {
  items.forEach((item, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      position: { left, top: top + i * gap + 8, width: 10, height: 10 },
      fill: C.blue,
      line: { style: "solid", fill: C.blue, width: 0 },
    });
    addText(slide, item, { left: left + 24, top: top + i * gap, width, height: 38 }, { fontSize, color: C.ink });
  });
}

function metric(slide, label, value, left, top, color = C.blue) {
  addBox(slide, { left, top, width: 250, height: 122, fill: C.white });
  addText(slide, String(value), { left: left + 22, top: top + 22, width: 210, height: 48 }, { fontSize: 38, bold: true, color });
  addText(slide, label, { left: left + 22, top: top + 76, width: 210, height: 24 }, { fontSize: 15, color: C.muted });
}

function simpleTable(slide, headers, rows, left, top, widths, rowH = 34) {
  const total = widths.reduce((a, b) => a + b, 0);
  addBox(slide, { left, top, width: total, height: rowH * (rows.length + 1), fill: C.white });
  let x = left;
  headers.forEach((h, i) => {
    slide.shapes.add({ geometry: "rect", position: { left: x, top, width: widths[i], height: rowH }, fill: C.blue2, line: { style: "solid", fill: C.line, width: 1 } });
    addText(slide, h, { left: x + 8, top: top + 7, width: widths[i] - 16, height: rowH - 8 }, { fontSize: 13, bold: true, color: C.ink });
    x += widths[i];
  });
  rows.forEach((row, r) => {
    x = left;
    row.forEach((cell, i) => {
      slide.shapes.add({ geometry: "rect", position: { left: x, top: top + rowH * (r + 1), width: widths[i], height: rowH }, fill: C.white, line: { style: "solid", fill: C.line, width: 1 } });
      addText(slide, String(cell), { left: x + 8, top: top + rowH * (r + 1) + 7, width: widths[i] - 16, height: rowH - 8 }, { fontSize: 12, color: C.ink });
      x += widths[i];
    });
  });
}

function slide1() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  addText(s, "VAST Challenge 2023 MC1", { left: 72, top: 92, width: 820, height: 56 }, { fontSize: 28, bold: true, color: C.blue });
  addText(s, "非法捕鱼知识图谱可视分析", { left: 72, top: 160, width: 900, height: 82 }, { fontSize: 54, bold: true, color: C.ink });
  addText(s, "基于 FishHook 思路的可疑实体扩展、异常评分与多视图分析", { left: 76, top: 270, width: 850, height: 40 }, { fontSize: 22, color: C.muted });
  addBox(s, { left: 78, top: 372, width: 1000, height: 150, fill: C.white });
  bulletList(s, ["从官方 4 个可疑实体出发", "追踪公司、船只、人物、组织之间的可疑关系", "用多指标可视化解释调查优先级"], 110, 402, 850, 22, 38);
  footer(s, 1);
}

function slide2() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "研究背景与任务", "MC1 的关键不是画完整大图，而是为 FishEye 找到值得继续调查的实体。");
  bulletList(s, [
    "数据来自文本抽取形成的知识图谱，包含实体和关系。",
    "官方给出 4 个可疑实体，要求分析它们的上下文并发现其他可疑对象。",
    "课程项目目标：构建可交互系统，并形成可解释的证据链。",
  ], 88, 150, 960, 23, 60);
  footer(s, 2);
}

function slide3() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "数据概况", "节点、边、实体类型和关系类型构成后续分析基础。");
  metric(s, "节点数量", data.node_count, 90, 160, C.blue);
  metric(s, "边数量", data.edge_count, 370, 160, C.red);
  metric(s, "官方可疑实体", 4, 650, 160, C.amber);
  addBox(s, { left: 90, top: 340, width: 980, height: 170, fill: C.white });
  bulletList(s, [
    "节点类型：company、person、organization、vessel、location 等。",
    "关系类型：ownership、membership、partnership、family_relationship。",
    "图结构：有向多重图，同一对实体之间可存在多条不同关系。",
  ], 122, 374, 850, 20, 38);
  footer(s, 3);
}

function slide4() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "系统总体设计", "当前项目对应 FishHook 的多视图分析思想。");
  const steps = [
    ["官方可疑实体", "调查种子"],
    ["Ego Net", "局部上下文"],
    ["候选扩展", "寻找新实体"],
    ["异常评分", "排序优先级"],
    ["多视图解释", "路径/共同邻居/平行坐标"],
  ];
  steps.forEach((step, i) => {
    const left = 78 + i * 226;
    addBox(s, { left, top: 230, width: 178, height: 110, fill: i === 0 ? C.red2 : C.white, line: i === 0 ? C.red : C.line });
    addText(s, step[0], { left: left + 14, top: 254, width: 150, height: 28 }, { fontSize: 21, bold: true, color: i === 0 ? C.red : C.ink });
    addText(s, step[1], { left: left + 14, top: 292, width: 150, height: 24 }, { fontSize: 15, color: C.muted });
    if (i < steps.length - 1) addText(s, "→", { left: left + 184, top: 264, width: 40, height: 40 }, { fontSize: 34, bold: true, color: C.blue });
  });
  footer(s, 4);
}

function slide5() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "FishHook 风格异常特征", "用多个结构指标近似论文中的异常模式检测。");
  simpleTable(s, ["指标", "含义"], [
    ["异常分", "综合异常程度，用于排序"],
    ["船只比例", "弱连通分量中 vessel 占比"],
    ["家族边", "1-2 跳 family_relationship 数量"],
    ["政治组织", "1-2 跳政治组织数量"],
    ["短环路", "复杂控制结构或供应链闭环线索"],
    ["关联嫌疑", "连接到几个官方可疑实体"],
    ["所有权/成员边", "控制结构与组织归属"],
  ], 100, 142, [190, 770], 44);
  footer(s, 5);
}

function slide6() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "平行坐标图怎么看", "每条线是一个候选实体，每根竖轴是一个异常特征。");
  addBox(s, { left: 86, top: 150, width: 1030, height: 340, fill: C.white });
  const axes = ["异常分", "船只比例", "家族边", "政治组织", "短环路", "关联嫌疑", "船只连接", "所有权边", "成员边"];
  axes.forEach((axis, i) => {
    const x = 130 + i * 110;
    s.shapes.add({ geometry: "line", position: { left: x, top: 205, width: 0, height: 210 }, line: { style: "solid", fill: C.line, width: 2 } });
    addText(s, axis, { left: x - 35, top: 170, width: 90, height: 24 }, { fontSize: 13, bold: true, color: C.ink });
  });
  const lineSets = [[0.92,0.1,0.85,0.9,0.8,1,0.78,0.9,0.88],[0.45,0.1,0.5,0.3,0.1,0.5,0.3,0.2,0.35],[0.72,0.1,0.66,0.55,0.45,1,0.6,0.5,0.55]];
  lineSets.forEach((vals, idx) => {
    for (let i=0; i<vals.length-1; i++) {
      const x1 = 130 + i*110, y1 = 415 - vals[i]*200;
      const x2 = 130 + (i+1)*110, y2 = 415 - vals[i+1]*200;
      s.shapes.add({ geometry: "line", position: { left: x1, top: y1, width: x2-x1, height: y2-y1 }, line: { style: "solid", fill: idx === 0 ? C.red : idx === 1 ? "#FCA5A5" : "#EF4444", width: idx === 0 ? 3 : 2 } });
    }
  });
  bulletList(s, ["线在多个关键轴上都靠上：说明候选实体多维度异常。", "只在单一轴上高：通常只能作为辅助线索。", "该图用于筛选优先调查对象，不直接证明违法。"], 110, 532, 950, 19, 34);
  footer(s, 6);
}

function slide7() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "FishHook 候选排名结果", "FishHook 分越高，越值得优先检查其关系网络与路径证据。");
  const rows = data.fishhook_top.slice(0, 6).map((r) => [
    String(r.candidate).slice(0, 26),
    r.type || "unknown",
    Number(r.fishhook_anomaly_score).toFixed(1),
    String(r.short_cycle_count ?? ""),
  ]);
  simpleTable(s, ["候选实体", "类型", "FishHook分", "短环路"], rows, 92, 148, [390, 170, 170, 140], 48);
  footer(s, 7);
}

function slide8() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "证据路径与共同邻居可视化", "用图解释候选实体为什么值得继续调查。");
  addBox(s, { left: 86, top: 150, width: 500, height: 360, fill: C.white });
  addText(s, "证据路径图", { left: 116, top: 178, width: 200, height: 30 }, { fontSize: 24, bold: true });
  bulletList(s, ["左到右阅读：当前实体 → 中间实体 → 关键词目标", "红色：官方可疑实体", "橙色：关键词相关目标", "线越粗：路径段重复越多"], 122, 235, 390, 18, 44);
  addBox(s, { left: 650, top: 150, width: 500, height: 360, fill: C.white });
  addText(s, "共同邻居图", { left: 680, top: 178, width: 220, height: 30 }, { fontSize: 24, bold: true });
  bulletList(s, ["热力图：共同邻居数量越多颜色越深", "网络图：可疑实体与共享中介的连接", "用于发现共同地点、组织或中介实体"], 686, 235, 390, 18, 48);
  footer(s, 8);
}

function slide9() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "实验结论", "当前系统能够支撑 MC1 的主要分析问题。");
  bulletList(s, [
    "979893388 等官方实体具有较强的关系网络复杂度。",
    "从官方可疑实体扩展，比单纯关键词匹配更符合调查逻辑。",
    "多指标同时偏高的候选对象适合作为 FishEye 的优先调查对象。",
    "证据路径和共同邻居可视化增强了报告中的证据链表达。",
  ], 92, 155, 980, 24, 66);
  footer(s, 9);
}

function slide10() {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  title(s, "不足与后续改进", "当前结果表示调查优先级，而不是违法定论。");
  bulletList(s, [
    "NLP 抽取的知识图谱存在乱码、匿名 ID 和关系噪声。",
    "短环路采用有界搜索，保证交互速度但不能穷举全部环路。",
    "关键词路径只是辅助线索，需要结合关系类型和原文语境。",
    "后续可加入图嵌入距离、社区发现、路径高亮和原文证据关联。",
  ], 92, 155, 980, 24, 62);
  footer(s, 10);
}

[slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8, slide9, slide10].forEach(fn => fn());

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

for (const [index, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(`${TMP}/${stem}.png`, await deck.export({ slide, format: "png", scale: 1 }));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${TMP}/${stem}.layout.json`, await layout.text());
}
await writeBlob(`${TMP}/deck-montage.webp`, await deck.export({ format: "webp", montage: true, scale: 1 }));
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(FINAL);
console.log(FINAL);
