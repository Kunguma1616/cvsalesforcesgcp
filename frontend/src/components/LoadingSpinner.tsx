import React from 'react';
import { createPortal } from 'react-dom';
import { colors } from '../config/colors';

interface LoadingSpinnerProps {
  className?: string;
  variant?: 'fixed' | 'absolute';
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  className = '',
  variant = 'fixed',
}) => {
  const animationStyles = `
    @keyframes slideWidth {
      0% {
        width: 0;
        opacity: 1;
      }
      50% {
        width: 100%;
        opacity: 1;
      }
      100% {
        width: 0;
        opacity: 1;
      }
    }
    .loading-bar-1 {
      animation: slideWidth 1.2s ease-in-out infinite;
    }
    .loading-bar-2 {
      animation: slideWidth 1.2s ease-in-out infinite 0.2s;
    }
    .loading-bar-3 {
      animation: slideWidth 1.2s ease-in-out infinite 0.4s;
    }
  `;

  const positionClass = variant === 'fixed' ? 'fixed' : 'absolute';

  return createPortal(
    <>
      <style>{animationStyles}</style>
      <div
        className={`${positionClass} inset-0 z-50 flex items-center justify-center pointer-events-none ${className}`}
        style={{ backgroundColor: 'rgba(255, 255, 255, 0.5)' }}
      >
        <div className="flex flex-col items-center justify-center">
          <div
            className="rounded-lg flex flex-col items-center justify-center gap-4 shadow-lg p-6"
            style={{
              backgroundColor: 'white',
              width: '160px',
              minHeight: '120px',
            }}
          >
            <div className="flex flex-col gap-2 mb-2" style={{ width: '48px' }}>
              <div
                className="loading-bar-1 h-1.5 rounded"
                style={{ backgroundColor: colors.primary.default }}
              ></div>
              <div
                className="loading-bar-2 h-1.5 rounded"
                style={{ backgroundColor: colors.primary.default }}
              ></div>
              <div
                className="loading-bar-3 h-1.5 rounded"
                style={{ backgroundColor: colors.primary.default }}
              ></div>
            </div>
            <p
              className="font-semibold text-sm"
              style={{
                color: colors.grayscale.body,
                fontFamily: 'Montserrat',
              }}
            >
              Processing...
            </p>
          </div>
        </div>
      </div>
    </>,
    document.body
  );
};

export default LoadingSpinner;
