import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import MessageComponent from "./components/Message";

export type Message = {
	id: string;
	text: string;
	sender: "me" | "server";
	time: string;
};

const WS_URL = "ws://localhost:8000/ws";

export default function App() {
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState("");
	const [connected, setConnected] = useState(false);
	const wsRef = useRef<WebSocket | null>(null);
	const listRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		const ws = new WebSocket(WS_URL);
		wsRef.current = ws;

		ws.addEventListener("open", () => {
			setConnected(true);
			setMessages((m) => [
				...m,
				{
					id: String(Date.now()),
					text: "Connected to server",
					sender: "server",
					time: new Date().toISOString(),
				},
			]);
		});

		ws.addEventListener("message", (ev) => {
			let text = "";
			try {
				const parsed = JSON.parse(ev.data);
				text = parsed.text ?? ev.data;
			} catch {
				text = ev.data;
			}
			setMessages((m) => [
				...m,
				{
					id: String(Date.now()) + Math.random(),
					text,
					sender: "server",
					time: new Date().toISOString(),
				},
			]);
		});

		ws.addEventListener("close", () => {
			setConnected(false);
			setMessages((m) => [
				...m,
				{
					id: String(Date.now()),
					text: "Disconnected",
					sender: "server",
					time: new Date().toISOString(),
				},
			]);
		});

		ws.addEventListener("error", () => {
			setConnected(false);
		});

		return () => {
			ws.close();
			wsRef.current = null;
		};
	}, []);

	useEffect(() => {
		// autoscroll to bottom
		if (listRef.current) {
			listRef.current.scrollTop = listRef.current.scrollHeight;
		}
	}, [messages]);

	function sendMessage() {
		if (!input.trim()) return;
		const msg = { query: input.trim() };
		// push locally
		setMessages((m) => [
			...m,
			{
				id: String(Date.now()) + Math.random(),
				text: input.trim(),
				sender: "me",
				time: new Date().toISOString(),
			},
		]);
		// send over WS
		try {
			wsRef.current?.send(JSON.stringify(msg));
		} catch {
			// fallback: try sending plain text
			wsRef.current?.send(input.trim());
		}
		setInput("");
	}

	function handleKey(e: React.KeyboardEvent<HTMLInputElement>) {
		if (e.key === "Enter") sendMessage();
	}

	const isLastMessage = (message: Message) => {
		const lastMessage = messages[messages.length - 1];
		if (!lastMessage) return false;
		return lastMessage.id === message.id;
	};

	return (
		<div id="root">
			<h1>Chat (WebSocket)</h1>
			<div className="chat-wrapper">
				<div className="chat-status">
					status: <strong>{connected ? "connected" : "disconnected"}</strong>
				</div>

				<div className="messages" ref={listRef} aria-live="polite">
					{messages.map((m) => (
						<MessageComponent
							key={m.id}
							message={m}
							isLast={isLastMessage(m)}
						/>
					))}
				</div>

				<div className="composer">
					<input
						value={input}
						onChange={(e) => setInput(e.target.value)}
						onKeyDown={handleKey}
						placeholder="Type a message and press Enter"
						aria-label="Message input"
					/>
					<button onClick={sendMessage}>Send</button>
				</div>
			</div>
		</div>
	);
}
