/**
 * Entry point: fetch the results summary, mount both surfaces.
 *
 * The model download is started but deliberately not awaited. The dashboard is
 * useful immediately, and the 24MB model only has to be ready by the time a
 * visitor actually drops a slice.
 */

import './styles.css';

import { loadClassifier } from './lib/inference';
import { bestNonPrivate, totalRuns, type ResultsSummary } from './lib/results';
import { formatPercent } from './lib/predictions';
import { mountChat } from './ui/chat';
import { mountDashboard } from './ui/dashboard';

const SUMMARY_URL = `${import.meta.env.BASE_URL}data/results_summary.json`;

function setText(id: string, text: string): void {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

async function fetchSummary(): Promise<ResultsSummary | null> {
  try {
    const response = await fetch(SUMMARY_URL);
    if (!response.ok) return null;
    return (await response.json()) as ResultsSummary;
  } catch {
    return null;
  }
}

async function start(): Promise<void> {
  const dashboardRoot = document.getElementById('dashboard-root');
  const chatRoot = document.getElementById('chat-root');

  const summary = await fetchSummary();

  if (dashboardRoot) {
    if (summary) {
      mountDashboard(dashboardRoot, summary);
    } else {
      const notice = document.createElement('p');
      notice.className = 'empty';
      notice.textContent =
        'The results summary could not be loaded. Rebuild it with "python -m src.aggregate".';
      dashboardRoot.append(notice);
    }
  }

  if (summary) {
    setText('fact-runs', String(totalRuns(summary)));
    setText('fact-chance', formatPercent(summary.chance_accuracy));
  }

  if (chatRoot) {
    mountChat(chatRoot, {
      classifier: loadClassifier(),
      referenceAccuracy: summary ? (bestNonPrivate(summary)?.accuracy_mean ?? null) : null,
    });
  }
}

void start();
