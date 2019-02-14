{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE OverloadedStrings #-}


module Net.MWE2 where


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
import qualified Net.ArcGraph as G


----------------------------------------------
-- Training dataset
----------------------------------------------


-- | Is MWE or not?
mwe, notMwe :: R 2
notMwe = LA.vector [1, 0]
mwe = LA.vector [0, 1]


-- | Training dataset
trainData :: FilePath -> IO (G.DataSet 300 2)
trainData path = do
  -- Load the embedding dictionary
  d <- D.load False path
  let vec = (d M.!)
  return $
    [ mkElem vec
        [ (0, "John")
        , (1, "made")
        , (2, "a")
        , (3, "mistake")
        ]
        [ (0, 1) =>> notMwe
        , (2, 3) =>> notMwe
        , (3, 1) =>> mwe
        ]
    , mkElem vec
        [ (0, "John")
        , (1, "made")
        , (2, "a")
        , (3, "boat")
        ]
        [ (0, 1) =>> notMwe
        , (2, 3) =>> notMwe
        , (3, 1) =>> notMwe
        ]
    , mkElem vec
        [ (0, "John")
        , (1, "made")
        , (2, "a")
        , (3, "chair")
        ]
        [ (0, 1) =>> notMwe
        , (2, 3) =>> notMwe
        , (3, 1) =>> notMwe
        ]
    , mkElem vec
        [ (0, "a")
        , (1, "cat")
        , (2, "made")
        , (3, "a")
        , (4, "nice")
        , (5, "mistake")
        ]
        [ (0, 1) =>> notMwe
        , (1, 2) =>> notMwe
        , (3, 5) =>> notMwe
        , (4, 5) =>> notMwe
        , (5, 2) =>> mwe
        ]
    ]
  where
    (=>>) e x = (e, x)
    mkElem vec nodes arcs =
      (graph, valMap)
      where
        vertices = [v | (v, _) <- nodes]
        gStr = Graph.buildG
          (minimum vertices, maximum vertices)
          (map fst arcs)
        lbMap = M.fromList [(v, vec h) | (v, h) <- nodes]
        graph = G.Graph
          { graphStr = gStr
          , graphInv = Graph.transposeG gStr
          , labelMap = lbMap }
        valMap = M.fromList arcs


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
  { iterNum = 250
  , gradient = BP.gradBP (G.netError dataSet depth)
  , quality = BP.evalBP (G.netError dataSet depth)
  , reportEvery = 10
  , gain0 = 0.01
  , tau = 100
  , gamma = 0.5
  }
