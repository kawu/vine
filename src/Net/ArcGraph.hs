{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE OverloadedStrings #-}
-- {-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TupleSections #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- To derive Binary for `Param`
{-# LANGUAGE DeriveAnyClass #-}

-- -- To make GHC automatically infer that `KnownNat d => KnownNat (d + d)`
-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fconstraint-solver-iterations=5 #-}


module Net.ArcGraph
  ( 
  -- * Network
--     Config(..)
    Param(..)
  -- , printParam
  -- , size
  , new
  , Graph(..)
  , Node(..)
  , Arc
  , run
  , eval
  , runQ
  , evalQ
  -- , runTest

  -- * Storage
  , saveParam
  , loadParam

  -- * Data set
  , DataSet
  , Elem(..)
  -- , mkGraph
  -- , mkSent

  -- * Error
  , netError
  , netErrorQ

  -- * Training
  -- , trainProg
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats

import           System.Random (randomRIO)

import           Control.Monad (forM_, forM)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
-- import qualified Data.List as L
import           Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Text as T
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as BL
import           Codec.Compression.Zlib (compress, decompress)

-- import qualified Data.Map.Lens as ML
import           Control.Lens.At (ixAt)
import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)
import qualified Numeric.SGD.ParamSet as SGD

import           Net.ArcGraph.Graph
import qualified Net.ArcGraph.BiComp as B
import qualified Net.ArcGraph.QuadComp as Q

-- import qualified Embedding.Dict as D

import           Debug.Trace (trace)


----------------------------------------------
-- Parameters
----------------------------------------------


-- -- | Parameter set
-- type Param dim a b 
--    = Biaff dim dim
--   :& Bias


-- -- | Parameter set (ONE)
-- type Param dim a b 
--    = Biaff dim dim
--   :& UnordBiaff dim dim
--   :& HeadAff dim dim
--   :& DepAff dim dim
--   :& HeadPosAff a
--   :& DepPosAff a
--   :& PapyPosAff a
--   :& EnkelPosAff a
--   :& ArcBias b
--   :& PapyArcAff b
--   :& EnkelArcAff b
--   :& Bias


-- -- | Parameter set (HOLISTIC)
-- type Param dim a b 
--    = Holi dim 50 a b 100 100
-- --   :& UnordBiaff dim dim
-- --   :& HeadAff dim dim
-- --   :& DepAff dim dim
-- --   :& HeadPosAff a
-- --   :& DepPosAff a
-- --   :& PapyPosAff a
-- --   :& EnkelPosAff a
-- --   :& ArcBias b
-- --   :& PapyArcAff b
-- --   :& EnkelArcAff b
--   :& Bias


-- -- | Parameter set (TWO; ONE without unordered binary relation)
-- type Param dim a b
--    = Biaff dim dim
--   -- :& UnordBiaff dim dim
--   :& HeadAff dim dim
--   :& DepAff dim dim
--   :& HeadPosAff a
--   :& DepPosAff a
--   :& PapyPosAff a
--   :& EnkelPosAff a
--   :& ArcBias b
--   :& PapyArcAff b
--   :& EnkelArcAff b
--   :& Bias


-- -- | quad100
-- type Param d a b
-- --    = Q.QuadAff d d
--    = Q.TriAff d d
--   :& Q.SibAff d d
--   :& Q.BiAff d d
--   :& Q.UnAff d d
--   :& Q.Bias


-- -- | quad100-unord
-- type Param d a b
--    = Q.TriAff d 100
--   :& Q.SibAff d 100
--   :& Q.BiAff d 100
--   :& Q.UnordBiAff d 100
--   :& Q.UnAff d 100
--   :& Q.Bias


-- -- | The selected parameter config
-- type Param d a b = Arc1 d a b


-- | Possible model configurations
data Param d a b 
  = PArc0 (Arc0 d a b)
  | PArc1 (Arc1 d a b)
  | PArc2 (Arc2 d a b)
  | PArc3 (Arc3 d a b)
  | PArc4 (Arc4 d a b)
  | PQuad0 (Quad0 d a b)
  | PQuad1 (Quad1 d a b)
  deriving (Generic, Binary)


-- | Arc-factored model (0)
type Arc0 d a b
   = Q.BiAff d 100
  :& Q.Bias


-- | Arc-factored model (1)
type Arc1 d a b
   = Q.BiAff d 100
  :& Q.BiQuad (BiParam a b)


-- | Arc-factored model (2)
type Arc2 d a b
   = Q.BiAffExt d 50 a b 100
  :& Q.BiQuad (BiParam a b)


-- | Arc-factored model (3)
type Arc3 d a b
  = Q.BiQuad (PotArc d 100 a b)


-- | Arc-factored model (4); similar to (3), but with (h = d).
-- The best for LVC.full in French found so far!
type Arc4 d a b
  = Q.BiQuad (PotArc d d a b)


-- | Basic bi-affine compoments (easy to compute, based on POS and DEP labels
-- exclusively)
type BiParam a b 
   = B.Bias
  :& B.HeadPosAff a
  :& B.DepPosAff a
  :& B.PapyPosAff a
  :& B.EnkelPosAff a
  :& B.ArcBias b
  :& B.PapyArcAff b
  :& B.EnkelArcAff b


-- | The best arc-factored model you found so far (with h = dim).
type PotArc dim h a b 
   = B.Biaff dim h
  :& B.UnordBiaff dim h
  :& B.HeadAff dim h
  :& B.DepAff dim h
  :& BiParam a b


-- | Quad-factored model (0) with hidden layers of size 100
type Quad0 d a b = 
  QuadH d 100 a b


-- | Quad-factored model (1); (0) + unordered bi-affine component.  That's the
-- last model you tried on cl-srv2.  Turned out better than `Quad0`, the
-- `Q.UnordBiAff` seems to make a difference.
--
-- Now you could try to do some ablation/enriching study:
--
--   * What if you remove `Q.TriAff` and `Q.SibAff`?
--   * What if you also remove `Q.UnAff`?
--
-- You can also try `Arc3`, a simplified (with dim = 100) version of the best
-- model obtained so far (`Arc4`).  Just to see how much you potentialy (well,
-- it can depend on training anyway...) lose by using smaller dimension.
--
-- The next thing to do would be to check if you can gain somethin by using
-- POS/DEP embeddings.  For instance by enriching `Arc3` or `Arc4` with
-- `Q.BiAffExt` (see `Arc2`).  In fact, it makes sense to test `Arc1`, `Arc2`,
-- and `Arc3` first and see what they give.
--
type Quad1 d a b
   = QuadH d 100 a b
  :& Q.UnordBiAff d 100


-- | Quad-factored model with underspecified size of the hidden layers
type QuadH d h a b
   = Q.TriAff d h
  :& Q.SibAff d h
  :& Q.BiAff d h
  :& Q.UnAff d h
  :& Q.Bias


----------------------------------------------
-- Evaluation
----------------------------------------------


-- printParam :: (Show a, Show b) => Param dim a b -> IO ()
-- printParam
--   ( biaff :& unordBiaff :& headAff :& depAff :&
--     headPosAff :& depPosAff :& papyPosAff :&
--     enkelPosAff :& arcBias :& papyArcAff :&
--     enkelArcAff :& bias
--   ) = do
--     print headPosAff
--     print depPosAff
--     print papyPosAff
--     print enkelPosAff
--     print arcBias
--     print papyArcAff
--     print enkelArcAff
--     print bias


run
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , B.BiComp dim a b comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = B.runBiComp graph arc net
  return (arc, logistic x)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
eval net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        graph
    )


runQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , Q.QuadComp dim a b comp
     )
  => BVar s comp
    -- ^ Quad component
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
runQ net graph = normalize . M.unionsWith (+) $ do
  arc <- graphArcs graph
  return $ Q.runQuadComp graph arc net
  where
    normalize = fmap logistic


evalQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , Q.QuadComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
evalQ net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runQ
        (BP.constVar net)
        graph
    )


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam :: (Binary a) => FilePath -> a -> IO ()
saveParam path =
  BL.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam :: (Binary a) => FilePath -> IO a
loadParam path =
  Bin.decode . decompress <$> BL.readFile path
  -- BL.writeFile path . compress . Bin.encode


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim a b = Elem
  { graph :: Graph (Node dim a) b
    -- ^ Input graph
  , labMap :: M.Map Arc Double
    -- ^ Target probabilities
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of (graph, target value map) pairs.
type DataSet d a b = [Elem d a b]


----------------------------------------------
-- Error
----------------------------------------------


-- -- | Squared error between the target and the actual output.
-- errorOne
--   :: (Ord a, KnownNat n, Reifies s W)
--   => M.Map a (BVar s (R n))
--     -- ^ Target values
--   -> M.Map a (BVar s (R n))
--     -- ^ Output values
--   -> BVar s Double
-- errorOne target output = PB.sum . BP.collectVar $ do
--   (key, tval) <- M.toList target
--   let oval = output M.! key
--       err = tval - oval
--   return $ err `dot` err


-- | Cross entropy between the true and the artificial distributions
crossEntropy
  :: (Reifies s W)
  => BVar s Double
    -- ^ Target ,,true'' MWE probability
  -> BVar s Double
    -- ^ Output ,,artificial'' MWE probability
  -> BVar s Double
crossEntropy p q = negate
  ( p0 * log q0
  + p1 * log q1
  )
  where
    p1 = p
    p0 = 1 - p1
    q1 = q
    q0 = 1 - q1


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: (KnownNat n, Reifies s W)
--   => BVar s (R n)
--     -- ^ Target ,,true'' distribution
--   -> BVar s (R n)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropy p q =
--   negate (p `dot` LBP.vmap' log q)


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
--   => M.Map a (BVar s (R n))
  => M.Map a (BVar s Double)
    -- ^ Target values
--   -> M.Map a (BVar s (R n))
  -> M.Map a (BVar s Double)
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
--   => [M.Map a (BVar s (R n))] -- ^ Targets
--   -> [M.Map a (BVar s (R n))] -- ^ Outputs
  => [M.Map a (BVar s Double)] -- ^ Targets
  -> [M.Map a (BVar s Double)] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: ( Reifies s W, KnownNat dim
     , Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => DataSet dim a b
  -- -> BVar s (Param dim a b)
  -> BVar s comp
  -> BVar s Double
netError dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (run net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)


-- | Network error on a given dataset.
netErrorQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , Q.QuadComp dim a b comp
     )
  => DataSet dim a b
  -> BVar s comp
  -> BVar s Double
netErrorQ dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (runQ net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)
