{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE OverloadedStrings #-}


module Net.MWE where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.Monad (guard, forM_)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Ord (comparing)
import qualified Data.List as L
import qualified Data.Text as T
import qualified Data.Map.Strict as M
import qualified Data.Graph as Graph
import qualified Data.Array as A

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
import qualified GradientDescent as GD
import qualified GradientDescent.Momentum as Mom

import qualified Embedding.Dict as D
import qualified Net.Graph as G


----------------------------------------------
-- Training dataset
----------------------------------------------


-- | Is MWE or not?
mwe, notMwe :: R 2
mwe = LA.vector [1, 0]
notMwe = LA.vector [0, 1]


-- | Training dataset
trainData :: FilePath -> IO (G.Dataset 300 2)
trainData path = do
  -- Load the embedding dictionary
  d <- D.load path
  let vec = (d M.!)
  return $
    [ mkElem vec
        [ (0, "John") =>> notMwe 
        , (1, "made") =>> mwe
        , (2, "a") =>> notMwe
        , (3, "mistake") =>> mwe
        ]
        [(0, 1), (2, 3), (3, 1)]
    , mkElem vec
        [ (0, "John") =>> notMwe 
        , (1, "made") =>> notMwe
        , (2, "a") =>> notMwe
        , (3, "boat") =>> notMwe
        ]
        [(0, 1), (2, 3), (3, 1)]
    , mkElem vec
        [ (0, "John") =>> notMwe 
        , (1, "made") =>> notMwe
        , (2, "a") =>> notMwe
        , (3, "chair") =>> notMwe
        ]
        [(0, 1), (2, 3), (3, 1)]
    , mkElem vec
        [ (0, "a") =>> notMwe 
        , (1, "cat") =>> notMwe 
        , (2, "made") =>> mwe
        , (3, "a") =>> notMwe
        , (4, "nice") =>> notMwe
        , (5, "mistake") =>> mwe
        ]
        [(0, 1), (1, 2), (3, 5), (4, 5), (5, 2)]
    ]
  where
    (=>>) (v, h) x = (v, h, x)
    mkElem vec nodes arcs =
      (graph, valMap)
      where
        vertices = [v | (v, _, _) <- nodes]
        gStr = Graph.buildG (minimum vertices, maximum vertices) arcs
        lbMap = M.fromList [(v, vec h) | (v, h, _) <- nodes]
        graph = G.Graph
          { graphStr = gStr
          , graphInv = Graph.transposeG gStr
          , labelMap = lbMap }
        valMap = M.fromList [(v, x) | (v, _, x) <- nodes]


----------------------------------------------
-- Training
----------------------------------------------


-- | Train with the default dataset.
train embPath depth net0 = do
  dataSet <- trainData embPath
  G.trainProg (gdCfg dataSet) depth net0


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- | Gradient descent configuration
gdCfg dataSet depth = Mom.Config
  { iterNum = 200
  , gradient = BP.gradBP (G.netError dataSet depth)
  , quality = BP.evalBP (G.netError dataSet depth)
  , reportEvery = 10
  , gain0 = 0.05
  , tau = 100
  , gamma = 0.5
  }
