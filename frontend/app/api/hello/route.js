export async function POST(request) {
  try {
    const response = await fetch("http://127.0.0.1:8000/api/evaluate_tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: await request.text(),
    });

    const data = await response.json();
    return new Response(JSON.stringify(data), { status: 200 });
  } catch (error) {
    return new Response(JSON.stringify({ message: "Backend error" }), { status: 500 });
  }
}
