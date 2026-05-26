import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import App from './App';

describe('App', () => {
  it('renders upload controls', () => {
    render(<App />);

    expect(screen.getByText('Squat Analysis')).toBeInTheDocument();
    expect(screen.getByText('Drop or choose a squat video')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /process video/i })).toBeInTheDocument();
  });
});
